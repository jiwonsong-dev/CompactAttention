# Copyright (c) 2024, NVIDIA CORPORATION.  Adapted from NVIDIA RULER (https://github.com/NVIDIA/RULER), Apache-2.0 License.
# Licensed under The MIT License [see LICENSE for details]

"""
Prepare jsonl with field `input` and `outputs`.
{
    "index" int,
    "input": str,
    "outputs": [str],
}

python prepare.py \
    --save_dir ./ \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --model_template_type base \
    --num_samples 10 \
"""
import argparse
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml
from template import Templates

TASK_TEMPLATE_PLACEHOLDER = "__RULER_TASK_TEMPLATE_PLACEHOLDER__"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir", type=Path, required=True, help="dataset folder to save dataset"
)
parser.add_argument(
    "--benchmark", type=str, default="synthetic", help="Options: [synthetic]"
)
parser.add_argument("--task", type=str, required=True, help="tasks in benchmark")
parser.add_argument(
    "--subset", type=str, default="validation", help="Options: validation or test"
)
parser.add_argument(
    "--tokenizer_path", type=str, required=True, help="path to the tokenizer model"
)
parser.add_argument(
    "--tokenizer_type", type=str, default="nemo", help="[Options] nemo, hf, openai."
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    required=True,
    help="max sequence length including all input tokens and generated tokens.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=500,
    help="maximum number of samples we want to test",
)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument(
    "--model_template_type", type=str, default="base", help="Options in `template.py`"
)
parser.add_argument(
    "--chat_template_enable_thinking",
    type=str,
    default="auto",
    choices=["auto", "true", "false"],
    help=(
        "Only used when model_template_type resolves via a Hugging Face chat template. "
        "Set to false for Qwen3 non-reasoning prompts."
    ),
)
parser.add_argument(
    "--remove_newline_tab",
    action="store_true",
    help="remove `\n` and `\t` in all strings.",
)
parser.add_argument(
    "--chunk_idx", type=int, default=0, help="index of current split chunk"
)
parser.add_argument("--chunk_amount", type=int, default=1, help="size of split chunk")

args = parser.parse_args()


def resolve_model_template():
    if args.model_template_type in Templates:
        return Templates[args.model_template_type]

    if args.model_template_type not in {"hf-chat", "hf-chat-no-thinking"}:
        raise ValueError(
            f"{args.model_template_type} is not found in {list(Templates.keys())} "
            "or the supported dynamic chat template types "
            "['hf-chat', 'hf-chat-no-thinking']"
        )

    from transformers import AutoConfig, AutoTokenizer

    tokenizer_path = args.tokenizer_path
    try:
        config = AutoConfig.from_pretrained(tokenizer_path)
        tokenizer_path = getattr(config, "base_model", tokenizer_path)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    apply_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.model_template_type == "hf-chat-no-thinking":
        apply_kwargs["enable_thinking"] = False
    elif args.chat_template_enable_thinking == "true":
        apply_kwargs["enable_thinking"] = True
    elif args.chat_template_enable_thinking == "false":
        apply_kwargs["enable_thinking"] = False

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": TASK_TEMPLATE_PLACEHOLDER}],
        **apply_kwargs,
    )
    if TASK_TEMPLATE_PLACEHOLDER not in rendered:
        raise ValueError(
            "Dynamic chat template rendering lost the task placeholder; "
            f"template_type={args.model_template_type}"
        )
    return rendered.replace(TASK_TEMPLATE_PLACEHOLDER, "{task_template}")


def main():
    start_time = time.time()
    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])

    # Add templates
    print("Using model template:!!!!!!!!!!!!!!", args.model_template_type)
    model_template = resolve_model_template()
    task_template = config["template"]

    # Add answer prefix for all models
    answer_prefix = config["answer_prefix"] if "answer_prefix" in config else ""
    config["template"] = (
        model_template.format(task_template=task_template) + answer_prefix
    )

    # Split task into multiple chunks
    chunks = [
        (args.num_samples // args.chunk_amount)
        + (1 if i < args.num_samples % args.chunk_amount else 0)
        for i in range(args.chunk_amount)
    ]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[: args.chunk_idx])

    random_seed = 42 + args.chunk_idx

    try:
        script = os.path.join(curr_folder, args.benchmark, f"{config['task']}.py")
        additional_args = " ".join([f"--{k} {v}" for k, v in config["args"].items()])
        py_exec = shlex.quote(sys.executable)
        script_path = shlex.quote(script)
        command = f"""{py_exec} {script_path} \
        --save_dir  {args.save_dir} \
        --save_name {args.task} \
        --subset {args.subset} \
        --tokenizer_path {args.tokenizer_path} \
        --tokenizer_type {args.tokenizer_type} \
        --max_seq_length {args.max_seq_length} \
        --tokens_to_generate {config['tokens_to_generate']} \
        --num_samples {num_samples} \
        --random_seed {random_seed} \
        {additional_args} \
        {f"--remove_newline_tab" if args.remove_newline_tab else ""} \
        {f"--pre_samples {pre_samples}" if config['task'] == 'qa' else ""} \
        --template "{config['template']}"
        """
        print(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            print("Output:")
            print(result.stdout)
        else:
            print("Error:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)

    save_file = args.save_dir / args.task / f"{args.subset}.jsonl"
    if not save_file.exists():
        raise FileNotFoundError(
            f"{save_file} was not created while preparing {args.task}. "
            "Check the generator output above."
        )
    print(f"Prepare {args.task} with lines: {args.num_samples} to {save_file}")
    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    main()
