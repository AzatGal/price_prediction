import os
import warnings

import numpy as np
import pandas as pd
import torch

# from datasets import Dataset
from data.custom_dataset.custom_dataset import CustomDataset

from easydict import EasyDict
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OrdinalEncoder, PowerTransformer, QuantileTransformer,
                                   FunctionTransformer, StandardScaler, OneHotEncoder)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    ROOT_DIR, 'data',
    # 'custom_dataset'
    'sberbank_housing'
    # 'homesite_insurance'
)


def prepare_date(seed, k):
    # columns = EasyDict(
    #     num=['Общая площадь',
    #          'Жилая площадь',
    #          'Площадь кухни',
    #          'Этаж',
    #          'Этажей в доме',
    #          'Лифт пассажирский (кол-во)',
    #          'Лифт грузовой (кол-во)',
    #          'Количество комнат',
    #          'Высота потолков',
    #          'Кол-во раздельных санузлов'],
    #     cat=['Тип продажи',
    #          'Объект продажи',
    #          'Мусоропровод',
    #          'Парковка',
    #          'Тип дома',
    #          'Вид из окон',
    #          'Расстояние до метро',
    #          'Округ',
    #          'Район'],
    #     label=['Стоимость']
    # )
    # raw_data = EasyDict({
    #     dataset_type: pd.read_csv(os.path.join(path, f"{dataset_type}.csv"))
    #     for dataset_type in ('train', 'val', 'test')
    # })
    # raw_data = EasyDict({
    #     dataset_type: EasyDict({
    #         feature_type: raw_data[dataset_type][columns[feature_type]]
    #         for feature_type in ('num', 'cat', 'label')
    #     })
    #     for dataset_type in ('train', 'val', 'test')
    # })

    ids = {
        part: np.load(os.path.join(DATA_PATH, 'split-default', f'{part}_idx.npy'))
        for part in ('train', 'val', 'test')
    }
    raw_data = dict(
        x_num=np.load(os.path.join(DATA_PATH, 'X_num.npy')),
        x_cat=np.concatenate([
            np.load(os.path.join(DATA_PATH, 'X_cat.npy')),
            np.load(os.path.join(DATA_PATH, 'X_bin.npy'))
        ], axis=1),
        y=np.load(os.path.join(DATA_PATH, 'Y.npy')),
    )

    raw_data = {
        part: {
            feature: pd.DataFrame(raw_data[feature][ids[part]])
            for feature in ('x_num', 'x_cat', 'y')
        }
        for part in ids.keys()
    }

    cats = [
        raw_data['train']['x_cat'][i].value_counts()
        for i in range(raw_data['train']['x_cat'].shape[1])
    ]
    cats = [cat.index[cat > 10].astype(str).to_numpy() for cat in cats]
    n_cats = [len(cat) + 1 for cat in cats]

    # warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    # warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

    noise = np.random.RandomState(seed).normal(
        0.0, 1e-5, raw_data['train']['x_num'].shape
    ).astype(raw_data['train']['x_num'].dtypes)
    # n_bins = [
    #     max(2, min(100, np.unique(raw_data['train']['x_num'][i]).shape[0] // 16))
    #     for i in range(raw_data['train']['x_num'].shape[1])
    # ]
    n_embed = dict(
        # x_num=n_bins,
        x_num=raw_data['train']['x_num'].shape[1],
        x_cat=n_cats
    )

    data_transformers = dict(
        x_num=make_pipeline(
            QuantileTransformer(
                n_quantiles=max(min(raw_data['train']['x_num'].shape[0] // 30, 1000), 10),
                output_distribution='normal',
                random_state=seed
            ),
            FunctionTransformer(np.nan_to_num),
            FunctionTransformer(lambda x: x.astype(np.float32)),
            # FunctionTransformer(
            #     lambda x: x.fillna(raw_data['train']['x_num'].min() - 100)
            # ),  # nan - как отдельный эмбеддинг  .min() - 100   .quantile(0.5)
            # KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans'), # , subsample=len(raw_data.train.num)),
            # FunctionTransformer(lambda x: x.astype(np.int64))
        ).fit(raw_data['train']['x_num']),
        x_cat=make_pipeline(
            FunctionTransformer(lambda x: x.astype(str)),
            OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1),
            FunctionTransformer(lambda x: (x + 1).astype(np.int64)),
        ).fit(raw_data['train']['x_cat']),
        y=make_pipeline(
            StandardScaler(),
            # FunctionTransformer(lambda x: x.to_numpy().astype(np.int64)),
            FunctionTransformer(lambda x: x.astype(np.float32))
        ).fit(raw_data['train']['y'])
    )
    # Create datasets with proper keys: x_num, x_cat, target, label
    datasets_dict = {}
    for part in ids.keys():
        x_num = data_transformers['x_num'].transform(raw_data[part]['x_num'])
        x_cat = data_transformers['x_cat'].transform(raw_data[part]['x_cat'])
        target = data_transformers['y'].transform(raw_data[part]['y'])
        label = raw_data[part]['y'].to_numpy()  # original labels for metric calculation
        datasets_dict[part] = CustomDataset(
            x_num, x_cat, target, label, k,  # part == 'train'
        )
    
    return datasets_dict, data_transformers, n_embed

# cfg = EasyDict(
#     raw_data=raw_data,
#     processors=EasyDict(
#         num=make_pipeline(
#             QuantileTransformer(output_distribution='normal'),
#             FunctionTransformer(np.nan_to_num),
#             FunctionTransformer(lambda x: x.astype(np.float32)),
#             # FunctionTransformer(
#             #     lambda x: x.fillna(raw_data.train.num.min() - 100)
#             # ),  # nan - как отдельный эмбеддинг  .min() - 100   .quantile(0.5)
#             # KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans'), # , subsample=len(raw_data.train.num)),
#             # FunctionTransformer(lambda x: x.astype(np.int64))
#         ),
#         cat=make_pipeline(
#             FunctionTransformer(lambda x: x.astype('str')),
#             OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1),
#             FunctionTransformer(lambda x: (x + 1).astype(np.int64))
#         ),
#         target=make_pipeline(
#             FunctionTransformer(lambda x: x)
#             # OneHotEncoder(sparse_output=False),
#             # StandardScaler(),
#             # PowerTransformer(),
#             # QuantileTransformer(output_distribution='normal'),
#             # FunctionTransformer(lambda x: x.astype(np.float32))
#         )
#     )
# )


if __name__ == '__main__':
    print(cfg.data_transformer)
