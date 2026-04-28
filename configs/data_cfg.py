import os

import pandas as pd
from easydict import EasyDict
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer, QuantileTransformer

from data.data_processing import DataTransformer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.path = os.path.join(ROOT_DIR, 'data', 'datasets')

df = pd.read_csv(os.path.join(cfg.path, "train.csv"))

cfg.num_cfg = {
    'columns': ['Стоимость',
                'Общая площадь',
                'Жилая площадь',
                'Площадь кухни',
                'Этаж',
                'Этажей в доме',
                'Лифт пассажирский (кол-во)',
                'Лифт грузовой (кол-во)',
                'Количество комнат',
                'Высота потолков',
                'Кол-во раздельных санузлов'],
    # 'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'num_processor.pkl')
}
cfg.num_cfg['processor'] = QuantileTransformer(output_distribution='normal').fit(
    df[cfg.num_cfg['columns']].fillna(-1.0)
)
# KBinsDiscretizer(encode='ordinal', n_bins=128, strategy='kmeans'),


cfg.cat_cfg = {
    'columns': ['Тип продажи',
                'Объект продажи',
                'Мусоропровод',
                'Парковка',
                'Тип дома',
                'Вид из окон',
                'Расстояние до метро',
                'Округ',
                'Район'],
    # 'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'cat_processor.pkl')
}
cfg.cat_cfg['processor'] = OrdinalEncoder(
    encoded_missing_value=-1, handle_unknown='use_encoded_value', min_frequency=26, unknown_value=-1
).fit(
    df[cfg.cat_cfg['columns']]
)

cats = [
    len(c) for c in cfg.cat_cfg['processor'].categories_
]
inf_cats = [
    0 if c is None else len(c) - 1
    for c in cfg.cat_cfg['processor'].infrequent_categories_
]
cfg.n_embed_cat = [i - j for i, j in zip(cats, inf_cats)]


cfg.target_cfg = {
    'columns': ['Стоимость'],
    # 'path': os.path.join(ROOT_DIR, 'data', 'data_transformers', 'target_processor.pkl')
}
cfg.target_cfg['processor'] = PowerTransformer().fit(df[cfg.target_cfg['columns']].to_numpy())


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
