import os
from easydict import EasyDict
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer

from data.data_preprocessing import DataTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.target = 'num'  # cat num
cfg.smooth = False
cfg.path = os.path.join(ROOT, 'data', 'datasets')
cfg.data_transformer = DataTransformer(
    num_cfg={'processor': KBinsDiscretizer(encode='ordinal', n_bins=128, strategy='kmeans'),
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
             'path': os.path.join(ROOT, 'data', 'data_transformers', 'num_processor.pkl')},
    cat_cfg={'processor': OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value',
                                         min_frequency=26, unknown_value=-1),
             'columns': ['Тип продажи',
                         'Объект продажи',
                         'Мусоропровод',
                         'Парковка',
                         'Тип дома',
                         'Вид из окон',
                         'Расстояние до метро',
                         'Округ',
                         'Район'],
             'path': os.path.join(ROOT, 'data', 'data_transformers', 'cat_processor.pkl')},
    target_cfg={'processor': PowerTransformer(),
                'columns': ['Стоимость'],
                'path': os.path.join(ROOT, 'data', 'data_transformers', 'target_processor.pkl')},
    target=cfg.target
)
cfg.features = cfg.data_transformer.num_cols + cfg.data_transformer.cat_cols


if __name__ == '__main__':
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
