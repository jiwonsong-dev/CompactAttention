from functools import wraps

import torch


def autocast_custom_fwd(fn):
    return torch.amp.custom_fwd(device_type="cuda")(fn)


def contiguous(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        new_args = [arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
        new_kwargs = {
            key: (value.contiguous() if isinstance(value, torch.Tensor) else value)
            for key, value in kwargs.items()
        }
        return fn(*new_args, **new_kwargs)

    return wrapper
