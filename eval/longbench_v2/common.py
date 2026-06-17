import re


LONG_BENCH_V2_PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "{context}\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


ANSWER_RE = re.compile(
    r"(?:correct answer is|answer is|answer:|^)\s*\(?\b([A-D])\b\)?",
    flags=re.IGNORECASE | re.MULTILINE,
)


def build_prompt(item: dict) -> str:
    return LONG_BENCH_V2_PROMPT.format(
        context=str(item.get("context", "")).strip(),
        question=str(item.get("question", "")).strip(),
        choice_A=str(item.get("choice_A", "")).strip(),
        choice_B=str(item.get("choice_B", "")).strip(),
        choice_C=str(item.get("choice_C", "")).strip(),
        choice_D=str(item.get("choice_D", "")).strip(),
    )


def extract_answer(response: str) -> str | None:
    text = str(response or "").replace("*", "").strip()
    match = re.search(r"The correct answer is\s*\(?([A-D])\)?", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\(([A-D])\)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def truncate_prompt_middle(tokenizer, prompt: str, max_input_tokens: int | None) -> tuple[str, int, bool]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_len = len(token_ids)
    if max_input_tokens is None or max_input_tokens <= 0 or original_len <= max_input_tokens:
        return prompt, original_len, False
    half = int(max_input_tokens) // 2
    kept = token_ids[:half] + token_ids[-(int(max_input_tokens) - half) :]
    return tokenizer.decode(kept, skip_special_tokens=True), original_len, True
