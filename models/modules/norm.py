import torch
import torch.nn as nn
# import torch.nn.functional as F


class NormEnsemble(nn.Module):
    def __init__(self,
                 norm: str,
                 normalized_shape: int | tuple[int],
                 k: int,
                 elementwise_affine: bool = False
                 ) -> None:
        super().__init__()
        self.norm = getattr(nn, norm)(normalized_shape, elementwise_affine=False)
        if elementwise_affine:
            if isinstance(normalized_shape, int):
                normalized_shape = k, 1, normalized_shape
            elif isinstance(normalized_shape, tuple):
                normalized_shape = k, 1, *normalized_shape
            else:
                raise Exception

            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.weight is not None:
            x = x * self.weight
        return x
