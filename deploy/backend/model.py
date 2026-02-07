import torch
import torch.nn as nn
import yaml
import numpy as np
from typing import Dict, List, Tuple
import os


class RealEstateModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.network(x)


class Predictor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = yaml.safe_load(open(config_path, encoding='utf-8'))
        self.features_config = self.config['features']
        self.categorical_mappings = {}
        self.model = None
        self.input_size = None
        self._prepare_mappings()
        self._init_model()

    def _prepare_mappings(self):
        """Подготовка маппингов для категориальных признаков"""
        for name, cfg in self.features_config.items():
            if cfg['type'] == 'categorical':
                self.categorical_mappings[name] = {opt: i for i, opt in enumerate(cfg['options'])}

    def _init_model(self):
        """Инициализация модели с правильным размером входа"""
        # Считаем размер входа: числовые (1 каждый) + категориальные (кол-во категорий для one-hot)
        self.input_size = 0
        for name, cfg in self.features_config.items():
            if cfg['type'] == 'categorical':
                self.input_size += len(cfg['options'])
            else:
                self.input_size += 1

        self.model = RealEstateModel(
            self.input_size,
            self.config['model']['hidden_size'],
            self.config['model']['dropout']
        )

        # Загрузка весов если есть, иначе инициализация случайными весами (для демо)
        weights_path = "model_weights.pth"
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
        else:
            # Инициализация весов для демонстрации
            self._train_dummy()

        self.model.eval()

    def _train_dummy(self):
        """Создание dummy весов для демонстрации"""
        # Генерируем синтетические данные для "обучения"
        torch.manual_seed(42)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Синтетическое обучение для создания разумных весов
        for _ in range(1000):
            X = torch.randn(32, self.input_size)
            # Цена ~ площадь * 100000 + шум
            y = X[:, 0] * 100000 + torch.randn(32) * 50000 + 2000000

            optimizer.zero_grad()
            pred = self.model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        torch.save(self.model.state_dict(), "model_weights.pth")

    def preprocess(self, features: Dict) -> torch.Tensor:
        """Преобразование входных данных в тензор"""
        processed = []

        for name, cfg in self.features_config.items():
            value = features.get(name, cfg.get('default'))

            if cfg['type'] == 'categorical':
                # One-hot encoding
                encoding = [0.0] * len(cfg['options'])
                idx = self.categorical_mappings[name].get(value, 0)
                encoding[idx] = 1.0
                processed.extend(encoding)
            else:
                # Нормализация
                min_val, max_val = cfg['min'], cfg['max']
                normalized = (float(value) - min_val) / (max_val - min_val)
                processed.append(normalized)

        return torch.tensor(processed, dtype=torch.float32).unsqueeze(0)

    def predict(self, features: Dict) -> Dict:
        """Предсказание с доверительным интервалом"""
        with torch.no_grad():
            X = self.preprocess(features)
            prediction = self.model(X).item()

            # Добавляем базовую цену и масштабируем
            base_price = 50000  # базовая цена за м²
            area = float(features.get('area', 50))

            # Корректировка предсказания
            adjusted_price = abs(prediction) * 1000 + base_price * area

            # Расчет доверительного интервала (эмуляция)
            uncertainty = adjusted_price * 0.15  # 15% погрешность
            confidence_low = adjusted_price - uncertainty
            confidence_high = adjusted_price + uncertainty

            return {
                'predicted_price': round(adjusted_price, 2),
                'confidence_low': round(max(confidence_low, 100000), 2),
                'confidence_high': round(confidence_high, 2),
                'uncertainty_percent': 15.0,
                'price_per_m2': round(adjusted_price / area, 2) if area > 0 else 0
            }


# Глобальный экземпляр предиктора
predictor = Predictor()