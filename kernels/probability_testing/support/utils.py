import torch
import torchvision


def as_tensors(X, *rest):
    "Calls as_tensor on a bunch of args, all of the first's device and dtype."
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]


def pil_grid(X, **kwargs):
    return torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(X, **kwargs))


def maybe_squeeze(X, dim):
    "Like torch.squeeze, but don't crash if dim already doesn't exist."
    return torch.squeeze(X, dim) if dim < len(X.shape) else X
