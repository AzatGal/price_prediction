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


import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothEmbedding(nn.Module):
    """
    Эмбеддинги, у которых соседние индексы изначально имеют высокое косинусное сходство.
    Инициализируются случайно, затем сглаживаются гауссовым фильтром.
    """
    def __init__(self, num_embeddings, embedding_dim, sigma=1.0, kernel_size=5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._init_smooth_weights(sigma, kernel_size)

    def _init_smooth_weights(self, sigma, kernel_size):
        # 1. Заполняем случайными значениями (например, нормальным распределением)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=.02)

        # 2. Строим одномерное гауссово ядро
        half = kernel_size // 2
        x = torch.arange(-half, half + 1).float()
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        # Превращаем в фильтр для Conv1d с группами
        kernel = gauss.view(1, 1, -1).repeat(self.embedding.embedding_dim, 1, 1)

        # 3. Применяем свёртку вдоль оси индексов
        # weight имеет форму (num_embeddings, embedding_dim)
        weight_in = self.embedding.weight.t().unsqueeze(0)  # (1, embedding_dim, num_embeddings)
        weight_out = F.conv1d(weight_in, kernel, padding=half,
                              groups=self.embedding.embedding_dim)
        self.embedding.weight.data[:self.num_embeddings // 2] = weight_out.squeeze(0).t()[:self.num_embeddings // 2]

    def forward(self, indices):
        return self.embedding(indices)

# --------------------------------
# Пример использования и проверки
if __name__ == "__main__":
    num_embeddings = 32
    dim = 16
    smoothing = SmoothEmbedding(num_embeddings, dim, sigma=1.5, kernel_size=7)

    with torch.no_grad():
        vectors = smoothing(torch.arange(num_embeddings))  # все эмбеддинги
        normed = F.normalize(vectors, dim=1)
        # Косинусная матрица
        sim = normed @ normed.T

    # Покажем, что для близких индексов сходство выше
    print("Косинусное сходство для соседних индексов:")
    for offset in [0, 1, 2, 3, 5, 10, 15]:
        diag_mean = torch.mean(torch.diag(sim, offset))
        print(f"|i-j| = {offset}: {diag_mean:.4f}")
