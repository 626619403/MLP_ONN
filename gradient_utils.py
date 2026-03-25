import torch


def add_gradient_noise(tensor: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
    noise_std = max(float(noise_std), 0.0)
    if noise_std <= 0:
        return tensor

    grad_scale = tensor.detach().std(unbiased=False)
    if torch.isnan(grad_scale) or grad_scale <= 0:
        return tensor

    return tensor + torch.randn_like(tensor) * (grad_scale * noise_std)
