import os
import warnings

import numpy as np
import pandas as pd
from easydict import EasyDict
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer, QuantileTransformer, \
    FunctionTransformer

# from data.data_processing import DataTransformer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(ROOT_DIR, 'data', 'datasets')

columns = EasyDict(
    num=['Общая площадь',
         'Жилая площадь',
         'Площадь кухни',
         'Этаж',
         'Этажей в доме',
         'Лифт пассажирский (кол-во)',
         'Лифт грузовой (кол-во)',
         'Количество комнат',
         'Высота потолков',
         'Кол-во раздельных санузлов'],
    cat=['Тип продажи',
         'Объект продажи',
         'Мусоропровод',
         'Парковка',
         'Тип дома',
         'Вид из окон',
         'Расстояние до метро',
         'Округ',
         'Район'],
    target=['Стоимость']
)

raw_data = EasyDict(
    train=pd.read_csv(os.path.join(path, "train.csv")),
    valid=pd.read_csv(os.path.join(path, "valid.csv")),
    test=pd.read_csv(os.path.join(path, "test.csv"))
)

cats = [raw_data.train[col].value_counts() for col in columns.cat]
cats = [cat.index[cat > 26].to_numpy() for cat in cats]
# n_bins = [
#     min(128, int(0.6 * raw_data.train[col].nunique()))
#     for col in columns.num
# ]

# print(n_bins)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

processors = EasyDict(
    num=make_pipeline(
        # QuantileTransformer(output_distribution='normal'),
        # FunctionTransformer(np.nan_to_num),
        # FunctionTransformer(lambda x: x.astype(np.float32)),
        FunctionTransformer(lambda x: x.fillna(raw_data.train[columns.num].min() - 10)),
        KBinsDiscretizer(n_bins=128, encode='ordinal', strategy='quantile'), #, subsample=len(raw_data.train)),
        FunctionTransformer(lambda x: x.astype(int))
    ),
    cat=make_pipeline(
        FunctionTransformer(lambda x: x.astype('str')),
        OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1),
        FunctionTransformer(lambda x: (x + 1).astype(int))
    ),
    target=make_pipeline(
        PowerTransformer(),
        FunctionTransformer(lambda x: x.astype(np.float32))
    )
)

cfg = EasyDict(
    processors=processors,
    columns=columns,
    # raw_data=raw_data,
    datasets=EasyDict(
        train=EasyDict(
            x_num=processors.num.fit_transform(raw_data.train[columns.num]),
            x_cat=processors.cat.fit_transform(raw_data.train[columns.cat]),
            target=processors.target.fit_transform(raw_data.train[columns.target].to_numpy()),
            label=raw_data.train[columns.target].to_numpy(),
        ),
        valid=EasyDict(
            x_num=processors.num.transform(raw_data.valid[columns.num]),
            x_cat=processors.cat.transform(raw_data.valid[columns.cat]),
            target=processors.target.transform(raw_data.valid[columns.target].to_numpy()),
            label=raw_data.valid[columns.target].to_numpy(),
        ),
        test=EasyDict(
            x_num=processors.num.transform(raw_data.test[columns.num]),
            x_cat=processors.cat.transform(raw_data.test[columns.cat]),
            target=processors.target.transform(raw_data.test[columns.target].to_numpy()),
            label=raw_data.test[columns.target].to_numpy(),
        ),
    ),
    # n_embed_num=len(columns.num),
    n_embed_num=processors.num.steps[1][1].n_bins_.tolist(),
    n_embed_cat=[len(cat) + 1 for cat in cats],
)
# print(cfg.n_embed_num)
# print(processors.num.steps[1][1].bin_edges_)


# cfg.num_cfg['processor'] = QuantileTransformer(output_distribution='normal').fit(
#     df[cfg.num_cfg['columns']].fillna(-1.0)
# )
# KBinsDiscretizer(encode='ordinal', n_bins=128, strategy='kmeans'),


    # 'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'cat_processor.pkl')
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
#              'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'num_processor.pkl')},
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
#              'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'cat_processor.pkl')},
#     target_cfg={'processor': PowerTransformer(),
#                 'columns': ['Стоимость'],
#                 'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'target_processor.pkl')},
# )
# cfg.features = cfg.data_transformer.num_cols + cfg.data_transformer.cat_cols




if __name__ == '__main__':
    print(cfg.data_transformer)
