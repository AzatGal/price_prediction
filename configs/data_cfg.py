import os
import warnings

import numpy as np
import pandas as pd

from easydict import EasyDict
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer, QuantileTransformer, \
    FunctionTransformer

from configs.model_cfg import cfg as model_cfg

# from dataset.data_processing import DataTransformer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = '/kaggle/input/competitions/sberbank-russian-housing-market'
# os.path.join(
#     ROOT_DIR, 'dataset',
#     # 'apartment_dataset'
#     'sberbank_housing'
# )


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
#         # feature_type: raw_data[dataset_type][columns[feature_type]]
#         for feature_type in ('num', 'cat', 'label')
#     })
#     for dataset_type in ('train', 'val', 'test')
# })


ids = EasyDict({
    dataset_type: np.load(os.path.join(path, 'split-random-0', f'{dataset_type}_idx.npy'))
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
        # cat=raw_data.cat[ids[dataset_type]],
        # num=raw_data.num[ids[dataset_type]],
        # label=raw_data.label[ids[dataset_type]],
        feature_type: pd.DataFrame(raw_data[feature_type][ids[dataset_type]])
        for feature_type in ('num', 'cat', 'label')
    })
    for dataset_type in ('train', 'val', 'test')
})

cats = [
    raw_data.train.cat[col].value_counts()
    for col in range(raw_data.train.cat.shape[1])
] # columns.cat]
cats = [cat.index[cat > 10].to_numpy() for cat in cats] # 26
# cats = [
#     np.unique(raw_data.train.cat[:, i], return_counts=True)
#     for i in range(raw_data.train.cat.shape[1])
# ]
# cats = [item[0][item[1] > 10] for item in cats]

# print(n_bins)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

cfg = EasyDict(
    raw_data=raw_data,
    # n_num=len(columns.num),
    # n_cat=[len(c) + 1 for c in cats],
    # columns=columns,
    processors=EasyDict(
        num=make_pipeline(
            # # PowerTransformer(),
            # QuantileTransformer(output_distribution='normal'),
            # FunctionTransformer(np.nan_to_num),
            # FunctionTransformer(lambda x: x.astype(np.float32)),
            FunctionTransformer(
                lambda x: x.fillna(raw_data.train.num.min() - 100)
            ),  # nan - как отдельный эмбеддинг  .min() - 100   .quantile(0.5)
            KBinsDiscretizer(n_bins=128, encode='ordinal', strategy='kmeans'),
            FunctionTransformer(lambda x: x.astype(int))
        ),
        cat=make_pipeline(
            FunctionTransformer(lambda x: x.astype('str')),
            OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1),
            FunctionTransformer(lambda x: (x + 1).astype(int))
        ),
        target=make_pipeline(
            PowerTransformer(),
            # QuantileTransformer(output_distribution='normal'),
            FunctionTransformer(lambda x: x.astype(np.float32))
        )
    )
)

# # model_cfg.n_embed_num = cfg.raw_data.train.num.shape[1]
# model_cfg.n_embed_num = (
#     cfg.processors.num.steps[1][1].n_bins_.tolist()
#     if isinstance(cfg.processors.num.steps[1][1], KBinsDiscretizer)
#     else cfg.raw_data.train.num.shape[1]
# )
# model_cfg.n_embed_cat = [
#     len(cat) + 1 for cat in cfg.processors.cat.steps[1][1].categories_
# ]

# cfg = EasyDict(
#     processors=processors,
#     columns=columns,
#     # raw_data=raw_data,
#     datasets=EasyDict(
#         train=EasyDict(
#             x_num=processors.num.fit_transform(raw_data.train[columns.num]),
#             x_cat=processors.cat.fit_transform(raw_data.train[columns.cat]),
#             target=processors.target.fit_transform(raw_data.train[columns.target].to_numpy()),
#             label=raw_data.train[columns.target].to_numpy(),
#         ),
#         valid=EasyDict(
#             x_num=processors.num.transform(raw_data.valid[columns.num]),
#             x_cat=processors.cat.transform(raw_data.valid[columns.cat]),
#             target=processors.target.transform(raw_data.valid[columns.target].to_numpy()),
#             label=raw_data.valid[columns.target].to_numpy(),
#         ),
#         test=EasyDict(
#             x_num=processors.num.transform(raw_data.test[columns.num]),
#             x_cat=processors.cat.transform(raw_data.test[columns.cat]),
#             target=processors.target.transform(raw_data.test[columns.target].to_numpy()),
#             label=raw_data.test[columns.target].to_numpy(),
#         ),
#     ),
#     # n_embed_num=len(columns.num),
#     n_embed_num=processors.num.steps[1][1].n_bins_.tolist(),
#     n_embed_cat=[len(cat) + 1 for cat in processors.cat.steps[1][1].categories],
# )
# print(cfg.n_embed_num)
# print(processors.num.steps[1][1].bin_edges_)


# cfg.num_cfg['processor'] = QuantileTransformer(output_distribution='normal').fit(
#     df[cfg.num_cfg['columns']].fillna(-1.0)
# )
# KBinsDiscretizer(encode='ordinal', n_bins=128, strategy='kmeans'),


    # 'path': os.path.join(ROOT_DIR, 'dataset', 'data_transformers', 'cat_processor.pkl')
# cfg.cat_cfg['processor'] = OrdinalEncoder(
#     encoded_missing_value=-1, handle_unknown='use_encoded_value', min_frequency=26, unknown_value=-1
# ).fit(
#     df[cfg.cat_cfg['columns']]
# )

# cats = [
#     len(c) for c in cfg.cat_cfg['processor'].categories_
# ]
# inf_cats = [
#     0 if c is None else len(c) - 1
#     for c in cfg.cat_cfg['processor'].infrequent_categories_
# ]
# cfg.n_embed_cat = [i - j for i, j in zip(cats, inf_cats)]


# cfg.target_cfg['processor'] = PowerTransformer().fit(df[cfg.target_cfg['columns']].to_numpy())


# cfg.data_transformer = DataTransformer(
#     num_cfg={'processor': KBinsDiscretizer(encode='ordinal', n_bins=128, strategy='kmeans'),
#              'columns': ['Стоимость',
#                          'Общая площадь',
#                          'Жилая площадь',
#                          'Площадь кухни',
#                          'Этаж',
#                          'Этажей в доме',
#                          'Лифт пассажирский (кол-во)',
#                          'Лифт грузовой (кол-во)',
#                          'Количество комнат',
#                          'Высота потолков',
#                          'Кол-во раздельных санузлов'],
#              'path': os.path.join(ROOT_DIR, 'dataset', 'data_transformers', 'num_processor.pkl')},
#     cat_cfg={'processor': OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value',
#                                          min_frequency=26, unknown_value=-1),
#              'columns': ['Тип продажи',
#                          'Объект продажи',
#                          'Мусоропровод',
#                          'Парковка',
#                          'Тип дома',
#                          'Вид из окон',
#                          'Расстояние до метро',
#                          'Округ',
#                          'Район'],
#              'path': os.path.join(ROOT_DIR, 'dataset', 'data_transformers', 'cat_processor.pkl')},
#     target_cfg={'processor': PowerTransformer(),
#                 'columns': ['Стоимость'],
#                 'path': os.path.join(ROOT_DIR, 'dataset', 'data_transformers', 'target_processor.pkl')},
# )
# cfg.features = cfg.data_transformer.num_cols + cfg.data_transformer.cat_cols




if __name__ == '__main__':
    print(cfg.data_transformer)
