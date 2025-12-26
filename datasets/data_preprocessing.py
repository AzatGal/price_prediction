import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer
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
        cat_processor: str,
        cat_processor_args: dict,
        num_processor: str,
        num_processor_args: dict,
        cat_columns: list,
        num_columns: list,
) -> ColumnTransformer:
    transformers = [
        (getattr(preprocessors, cat_processor)(**cat_processor_args), cat_columns),
        (getattr(preprocessors, num_processor)(**num_processor_args), num_columns)
    ]
    return ColumnTransformer(transformers, verbose=True, n_jobs=4)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    pass
