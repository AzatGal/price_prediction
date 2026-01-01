import os
import dill
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer  # , make_column_transformer
import sklearn.preprocessing as preprocessors


class DataTransformer:
    def __init__(self,
                 num_cfg,
                 cat_cfg,
                 target_cfg,
                 ) -> None:
        self._num_cfg = num_cfg
        self._cat_cfg = cat_cfg
        self._target_cfg = target_cfg

        self.num_processor = num_cfg['processor']
        self.cat_processor = cat_cfg['processor']
        self.target_processor = target_cfg['processor']

        self.num_cols = num_cfg['columns']
        self.cat_cols = cat_cfg['columns']
        self.target_cols = target_cfg['columns']

        if os.path.exists(num_cfg['path']):
            with open(num_cfg['path'], 'rb') as f:
                self.num_processor = dill.load(f)
        if os.path.exists(num_cfg['path']):
            with open(cat_cfg['path'], 'rb') as f:
                self.cat_processor = dill.load(f)
                self._set_cat_params()
        if os.path.exists(target_cfg['path']):
            with open(target_cfg['path'], 'rb') as f:
                self.target_processor = dill.load(f)

    def save(self):
        with open(self._num_cfg['path'], 'wb') as f:
            dill.dump(self.num_processor, f)
        with open(self._cat_cfg['path'], 'wb') as f:
            dill.dump(self.cat_processor, f)
        with open(self._target_cfg['path'], 'wb') as f:
            dill.dump(self.target_processor, f)

    def _set_cat_params(self):
        cats = [
            len(i) for i in self.cat_processor.categories_
        ]
        inf_cats = [
            0 if i is None else len(i)
            for i in self.cat_processor.infrequent_categories_
        ]
        self.num_categories = [i - j for i, j in zip(cats, inf_cats)]

    def fit(self, data):
        # data = data.fillna('-1')
        # print(data['Тип продажи'].unique())
        self.num_processor.fit(data[self.num_cols].fillna(-1.0))
        print(1)
        self.cat_processor.fit(data[self.cat_cols])
        #  .fillna('Пропущено значение'))
        print(2)
        self.target_processor.fit(data[self.target_cols])
        print(3)
        self._set_cat_params()

    def transform(self, data, is_target=False):
        # data = data.fillna('-1')
        if is_target:
            return self.target_processor.transform(data)
        else:
            num = self.num_processor.transform(data[self.num_cols].fillna(-1.0))
            cat = self.cat_processor.transform(data[self.cat_cols])
            #  .fillna('Пропущено значение'))
            for i, c in enumerate(self.num_categories):
                cat[cat[:, i] == -1, i] = c - 1
            return np.concat([num, cat], axis=1)

    def inverse_transform(self, data, is_target=False):
        if is_target:
            return self.target_processor.inverse_transform(data)
        else:
            num = self.num_processor.inverse_transform(data[self.num_cols])
            cat = self.cat_processor.inverse_transform(data[self.cat_cols])
            # for i, c in enumerate(self.amount_cats):
            #   cat[cat[:, i] == -1, i] = c
            # return np.concat([num, cat], axis=1)



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
