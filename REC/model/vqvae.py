import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single-codebook VQ module for VQ-VAE-style feature quantization.

    Input shape:  [..., dim]
    Output:
        quantized: same shape as input
        indices:   input shape without the last dim
        loss:      codebook loss + commitment loss
    """
    def __init__(self, dim, codebook_size=256, commitment_weight=0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight

        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        original_shape = x.shape
        flat_x = x.reshape(-1, self.dim)  # [N, D]

        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_x @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(original_shape)

        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(quantized.detach(), x)
        loss = codebook_loss + self.commitment_weight * commitment_loss

        quantized = x + (quantized - x).detach()

        indices = indices.view(original_shape[:-1])
        return quantized, indices, loss