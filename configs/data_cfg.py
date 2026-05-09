import os
import warnings

import numpy as np
import pandas as pd

from easydict import EasyDict
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer, \
                                   QuantileTransformer, FunctionTransformer, StandardScaler


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(
    ROOT_DIR, 'dataset',
    # 'apartment_dataset'
    'sberbank_housing'
)


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


ids = EasyDict({
    dataset_type: np.load(os.path.join(path, 'split-default', f'{dataset_type}_idx.npy'))
    for dataset_type in ('train', 'val', 'test')
})
raw_data = EasyDict(
    cat=np.concatenate(
        [
            np.load(os.path.join(path, 'X_cat.npy')),
            np.load(os.path.join(path, 'X_bin.npy'))
        ],
        axis=1
    ),
    num=np.load(os.path.join(path, 'X_num.npy')),
    label=np.load(os.path.join(path, 'Y.npy')),
)
raw_data = EasyDict({
    dataset_type: EasyDict({
        feature_type: pd.DataFrame(raw_data[feature_type][ids[dataset_type]])
        for feature_type in ('num', 'cat', 'label')
    })
    for dataset_type in ('train', 'val', 'test')
})

cats = [
    raw_data.train.cat[col].value_counts()
    for col in
    range(raw_data.train.cat.shape[1])
    # columns.cat
]
cats = [cat.index[cat > 10].to_numpy() for cat in cats]  # 26


warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

cfg = EasyDict(
    raw_data=raw_data,
    # n_num=len(columns.num),
    # n_cat=[len(c) + 1 for c in cats],
    processors=EasyDict(
        num=make_pipeline(
            # # PowerTransformer(),
            # QuantileTransformer(output_distribution='normal'),
            # FunctionTransformer(np.nan_to_num),
            # FunctionTransformer(lambda x: x.astype(np.float32)),
            FunctionTransformer(
                lambda x: x.fillna(raw_data.train.num.min() - 100)
            ),  # nan - как отдельный эмбеддинг  .min() - 100   .quantile(0.5)
            KBinsDiscretizer(n_bins=128, encode='ordinal', strategy='quantile'),
            FunctionTransformer(lambda x: x.astype(int))
        ),
        cat=make_pipeline(
            FunctionTransformer(lambda x: x.astype('str')),
            OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1),
            FunctionTransformer(lambda x: (x + 1).astype(int))
        ),
        target=make_pipeline(
            StandardScaler(),
            # PowerTransformer(),
            # QuantileTransformer(output_distribution='normal'),
            FunctionTransformer(lambda x: x.astype(np.float32))
        )
    )
)


if __name__ == '__main__':
    print(cfg.data_transformer)
