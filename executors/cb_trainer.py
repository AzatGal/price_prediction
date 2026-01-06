import os
import dill
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_percentage_error


def fit_catboost(train_dataset, val_dataset):
    cat_features = [
        'Тип продажи',
        'Объект продажи',
        'Мусоропровод',
        'Парковка',
        'Тип дома',
        'Вид из окон',
        'Расстояние до метро',
        'Округ',
        'Район'
    ]
    with open('/Users/azatgalautdinov/PycharmProjects/price_prediction/data/data_transformers/target_processor.pkl', 'rb') as f:
        target_processor = dill.load(f)
    train_target = target_processor.transform(train_dataset[['Стоимость']])
    val_target = target_processor.transform(val_dataset[['Стоимость']])
    label = val_dataset[['Стоимость']]

    # train_target = train_dataset[['Стоимость']]  # target_processor.transform()
    # val_target = val_dataset[['Стоимость']]  # target_processor.transform()

    train_dataset = train_dataset.drop(columns=['Стоимость'])
    train_dataset[cat_features] = train_dataset[cat_features].fillna('Нет значения')
    val_dataset = val_dataset.drop(columns=['Стоимость'])
    val_dataset[cat_features] = val_dataset[cat_features].fillna('Нет значения')

    train_pool = Pool(train_dataset, train_target,
                      cat_features=cat_features)
    val_pool = Pool(val_dataset, val_target,
                    cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=20_000,
        learning_rate=0.05,
        depth=10,
        loss_function='MAE',
        # eval_metric='MAPE',  # можно заменить на нужную метрику или оставить только loss
        verbose=1000,
        early_stopping_rounds=200,
        random_seed=42,
        l2_leaf_reg=1
    ).fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        # plot=False  # можно поставить True для визуализации в Jupyter
    )

    pred = target_processor.inverse_transform(model.predict(val_pool).reshape(-1, 1))
    print('MAPE: ', mean_absolute_percentage_error(label, pred))

    feature_importance = sorted(
        list(
            zip(train_dataset.columns, model.get_feature_importance())
        ),
        key=lambda x: x[1]
    )[::-1]

    print(pd.DataFrame(feature_importance, columns=['Признак', 'Важность']))

    return model


if __name__ == '__main__':
    path = '/Users/azatgalautdinov/PycharmProjects/price_prediction/data/datasets'
    train_dataset = pd.read_csv(os.path.join(path, 'train.csv')).drop(columns=['Unnamed: 0'])
    val_dataset = pd.read_csv(os.path.join(path, 'valid.csv')).drop(columns=['Unnamed: 0'])
    fit_catboost(train_dataset, val_dataset)
    # print(train_dataset)

"""
0:	learn: 0.7421810	test: 0.7371102	best: 0.7371102 (0)	total: 68.4ms	remaining: 22m 48s
1000:	learn: 0.0718784	test: 0.1038959	best: 0.1038959 (1000)	total: 34s	remaining: 10m 45s
2000:	learn: 0.0582226	test: 0.0990088	best: 0.0990088 (2000)	total: 1m 9s	remaining: 10m 25s
3000:	learn: 0.0530162	test: 0.0974451	best: 0.0974450 (2999)	total: 1m 45s	remaining: 9m 56s
4000:	learn: 0.0495079	test: 0.0964105	best: 0.0964105 (4000)	total: 2m 22s	remaining: 9m 29s
5000:	learn: 0.0470900	test: 0.0956858	best: 0.0956838 (4996)	total: 2m 57s	remaining: 8m 51s
6000:	learn: 0.0451075	test: 0.0950896	best: 0.0950896 (6000)	total: 3m 31s	remaining: 8m 14s
7000:	learn: 0.0437175	test: 0.0946929	best: 0.0946890 (6990)	total: 4m 6s	remaining: 7m 37s
8000:	learn: 0.0424693	test: 0.0942856	best: 0.0942856 (8000)	total: 4m 40s	remaining: 7m 1s
9000:	learn: 0.0412132	test: 0.0939539	best: 0.0939539 (9000)	total: 5m 15s	remaining: 6m 25s
10000:	learn: 0.0403156	test: 0.0937270	best: 0.0937269 (9999)	total: 5m 52s	remaining: 5m 52s
11000:	learn: 0.0394307	test: 0.0934686	best: 0.0934686 (11000)	total: 6m 28s	remaining: 5m 17s
12000:	learn: 0.0386447	test: 0.0932783	best: 0.0932770 (11994)	total: 7m 3s	remaining: 4m 42s
13000:	learn: 0.0378554	test: 0.0931053	best: 0.0931009 (12958)	total: 7m 39s	remaining: 4m 7s
14000:	learn: 0.0371141	test: 0.0928788	best: 0.0928788 (13992)	total: 8m 14s	remaining: 3m 31s
Stopped by overfitting detector  (200 iterations wait)

bestTest = 0.09281683573
bestIteration = 14420

Shrink modules to first 14421 iterations.
/Users/azatgalautdinov/PycharmProjects/price_prediction/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but PowerTransformer was fitted with feature names
  warnings.warn(
MAPE:  0.05062643910621021
                       Признак   Важность
0                Общая площадь  24.148380
1                        Округ  22.492477
2          Расстояние до метро   9.343367
3                        Район   9.277808
4                     Парковка   6.777528
5                Этажей в доме   4.112154
6              Высота потолков   3.621754
7                  Вид из окон   2.885054
8                Площадь кухни   2.737452
9                     Тип дома   2.518449
10                        Этаж   2.488714
11                 Тип продажи   2.463392
12               Жилая площадь   2.102312
13           Количество комнат   1.345585
14  Лифт пассажирский (кол-во)   1.257071
15      Лифт грузовой (кол-во)   1.232516
16  Кол-во раздельных санузлов   0.996658
17              Объект продажи   0.185637
18                Мусоропровод   0.013693

"""