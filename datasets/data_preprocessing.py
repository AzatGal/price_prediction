import pandas as pd

from sklearn.compose import ColumnTransformer  # , make_column_transformer
import sklearn.preprocessing as preprocessors


columns = [
    'Тип продажи',
    'Объект продажи',
    'Общая площадь',
    'Жилая площадь',
    'Площадь кухни',
    'Этаж',
    'Этажей в доме',
    'Лифт пассажирский (кол-во)',
    'Лифт грузовой (кол-во)',
    'Год строительства',
    'Балкон/лоджия',
    'Мусоропровод',
    'Парковка',
    'Количество комнат',
    'Тип дома',
    'Высота потолков',
    'Кол-во раздельных санузлов',
    'Кол-во совмещенных санузлов',
    'Ремонт',
    'Вид из окон',
    'Расстояние до метро',
    'Кол-во квартир в доме',
    'Адрес',
    'Стоимость',
    'Регион Циан'
]


def get_data_transformer(
        con_features_cfg: dict[str, any],
        cat_features_cfg: dict[str, any],
) -> ColumnTransformer:
    transformers = [
        (
            getattr(preprocessors, con_features_cfg['processor'])(**con_features_cfg['args']),
            con_features_cfg['columns']
        ),
        (
            getattr(preprocessors, cat_features_cfg['processor'])(**cat_features_cfg['args']),
            cat_features_cfg['columns']
        )
    ]
    return ColumnTransformer(transformers, verbose=True, n_jobs=4)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    pass
