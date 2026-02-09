import pandas as pd
import torch
import numpy as np
from typing import Dict, List
from easydict import EasyDict

from data.data_processing import DataTransformer
from models.transformers import TablePredictor


class PricePredictor:
    """Класс для загрузки модели и предсказаний"""
    def __init__(self,
                 model_cfg: EasyDict,
                 model_path: str,
                 features_keys: List[str],
                 data_transformer: DataTransformer,
                 error_margin: float,
                 inflation: float
                 ) -> None:
        self.model = TablePredictor(**model_cfg)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.features_keys = features_keys
        self.data_transformer = data_transformer
        self.error_margin = error_margin
        self.inflation = inflation

    @torch.no_grad()
    def predict(self, data: pd.DataFrame) -> Dict:
        features = torch.as_tensor(
            self.data_transformer.transform(data)
        )[:, 1:]
        # print(features)
        pred = self.model(features).numpy()
        price = self.data_transformer.inverse_transform(pred, target='num')
        price = self.inflation * price.item()
        lower_bound = price * (1 - self.error_margin)
        upper_bound = price * (1 + self.error_margin)
        res = {
            'predicted_price': int(price),
            'lower_bound': int(lower_bound),
            'upper_bound': int(upper_bound),
            'confidence': f"{(1 - self.error_margin) * 100:.0f}%",
            'error_margin': f"{self.error_margin * 100:.0f}%"
        }
        # print(res)
        return res

    def get_feature_list(self) -> List[Dict]:
        features = []
        for i in range(1, len(self.data_transformer.num_cols)):
            feature = dict(
                key=self.features_keys[i - 1],
                name=self.data_transformer.num_cols[i],
                type='number',
            )
            if 'площадь' in self.data_transformer.num_cols[i].lower():
                feature['unit'] = 'м²'
            if 'высота' in self.data_transformer.num_cols[i].lower():
                feature['unit'] = 'м'
            if np.any(self.data_transformer.num_processor.bin_edges_[i] < 0):
                feature['required'] = False
                feature['min'] = int(self.data_transformer.num_processor.bin_edges_[i][1])
            else:
                feature['required'] = True
                feature['min'] = int(self.data_transformer.num_processor.bin_edges_[i][0])
            features.append(feature)
        for i in range(len(self.data_transformer.cat_cols)):
            feature = dict(
                key=self.features_keys[len(self.data_transformer.num_cols) + i - 1],
                name=self.data_transformer.cat_cols[i],
                type='select',
            )
            options = self.data_transformer.cat_processor.categories_[i]
            mask = pd.isna(pd.Series(options))
            if mask.any():
                options[mask] = 'Другое'
                feature['required'] = False
            else:
                feature['required'] = True
            feature['options'] = options.tolist()
            features.append(feature)
        return features


# Singleton instance
_predictor = None


def get_predictor() -> PricePredictor:
    """Возвращает singleton экземпляр предиктора"""
    from configs.data_cfg import cfg as data_cfg
    from configs.model_cfg import cfg as model_cfg

    model_cfg.pred_dim = 1
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor(
            model_cfg=model_cfg,
            model_path='/Users/azatgalautdinov/Desktop/price_prediction/v1/PricePrediction.pt',
            features_keys=['area', 'living_area', 'kitchen_area', 'floor', 'total_floors',
                           'passenger_elevators', 'cargo_elevators', 'rooms', 'ceiling_height',
                           'separate_bathrooms', 'sale_type', 'object_type', 'garbage_chute',
                           'parking', 'house_type', 'window_view', 'distance_to_metro',
                           'district', 'neighborhood'],
            # features_keys=list(PredictionInput().dict().keys()),
            data_transformer=data_cfg.data_transformer,
            inflation=1,
            error_margin=0.05
        )
    return _predictor
