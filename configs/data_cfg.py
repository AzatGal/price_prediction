from easydict import EasyDict

data_cfg = EasyDict()
data_cfg.columns = [
    'Стоимость',  # 1111!!!!!!!!
    'Тип продажи',
    'Объект продажи',
    'Общая площадь',
    'Жилая площадь',
    'Площадь кухни',
    'Этаж',
    'Этажей в доме',
    'Лифт пассажирский (кол-во)',
    'Лифт грузовой (кол-во)',
    'Мусоропровод',
    'Парковка',
    'Количество комнат',
    'Тип дома',
    'Высота потолков',
    'Кол-во раздельных санузлов',
    'Вид из окон',
    'Расстояние до метро',
    'Адрес',
    'Регион Циан'
]
data_cfg.



(make_pipeline(
         SimpleImputer(fill_value=-1, strategy="constant"),
         KBinsDiscretizer(n_bins=128, encode='ordinal', strategy='kmeans')
     ), make_column_selector(dtype_include=np.number)(train)),
    (make_pipeline(
         SimpleImputer(fill_value='-1', strategy="constant"),
         OrdinalEncoder(min_frequency=26, handle_unknown='use_encoded_value', unknown_value=-1)
     ), make_column_selector(dtype_include=object)(train))