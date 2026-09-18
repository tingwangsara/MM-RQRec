import torch
from torch import nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim, num_codebooks=2, codebook_size=256, commitment_weight=0.25):
        super().__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim)
            for _ in range(num_codebooks)
        ])

        for codebook in self.codebooks:
            nn.init.normal_(codebook.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: [..., D]
        return:
            quantized_sum: [..., D]
            indices: [..., num_codebooks]
            rqvae_loss: scalar
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        residual = x_flat
        quantized_sum = torch.zeros_like(x_flat)
        all_indices = []
        rqvae_loss = 0.0

        for codebook in self.codebooks:
            code = codebook.weight  # [K, D]

            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ code.t()
                + code.pow(2).sum(dim=1).unsqueeze(0)
            )

            indices = torch.argmin(dist, dim=1)
            quantized = codebook(indices)

            # VQ loss
            codebook_loss = F.mse_loss(quantized, residual.detach())
            commitment_loss = F.mse_loss(quantized.detach(), residual)
            rqvae_loss = rqvae_loss + codebook_loss + self.commitment_weight * commitment_loss

            # straight-through estimator
            quantized_st = residual + (quantized - residual).detach()

            quantized_sum = quantized_sum + quantized_st
            residual = residual - quantized.detach()

            all_indices.append(indices)

        quantized_sum = quantized_sum.view(*original_shape)
        indices = torch.stack(all_indices, dim=-1).view(*original_shape[:-1], self.num_codebooks)

        return quantized_sum, indices, rqvae_loss