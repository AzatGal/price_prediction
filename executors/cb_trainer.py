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
    with open(
            '/Users/azatgalautdinov/PycharmProjects/price_prediction/data/data_transformers/target_processor.pkl',
            'rb'
    ) as f:
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

    # def mape(pred, label):
    #     pred = target_processor.inverse_transform(pred.reshape(-1, 1))
    #     label = target_processor.inverse_transform(label.reshape(-1, 1))
    #     return mean_absolute_percentage_error(label, pred)

    model = CatBoostRegressor(
        iterations=20_000,
        learning_rate=0.1,
        depth=10,
        loss_function='Huber:delta=8',  # Huber:delta=2 LogCosh MAE
        grow_policy='Depthwise',  # Lossguide Depthwise
        eval_metric='MAE',
        # custom_metric=mape,
        verbose=1000,
        early_stopping_rounds=100,
        random_seed=42,
        # bagging_temperature=1,
        # l2_leaf_reg=1
    ).fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        # plot=False  # можно поставить True для визуализации в Jupyter
    )

    # model.save_model('price_prediction.cbm')

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
    # print(train_dataset[['Стоимость', 'Этаж', 'Этажей в доме', 'Общая площадь', 'Высота потолков']].corr())

"""
Huber:delta=8
0:	learn: 0.7027768	test: 0.6971142	best: 0.6971142 (0)	total: 107ms	remaining: 35m 46s
1000:	learn: 0.0398214	test: 0.0865541	best: 0.0865541 (1000)	total: 26.2s	remaining: 8m 16s
2000:	learn: 0.0229365	test: 0.0829103	best: 0.0829036 (1996)	total: 49.5s	remaining: 7m 25s
3000:	learn: 0.0154898	test: 0.0820407	best: 0.0820388 (2986)	total: 1m 13s	remaining: 6m 56s
Stopped by overfitting detector  (100 iterations wait)

bestTest = 0.08183683772
bestIteration = 3670

Shrink model to first 3671 iterations.
/Users/azatgalautdinov/PycharmProjects/price_prediction/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but PowerTransformer was fitted with feature names
  warnings.warn(
MAPE:  0.044989731784954086
                       Признак   Важность
0                        Район  34.544076
1                Общая площадь  31.027129
2                        Округ  23.570381
3          Расстояние до метро   2.948685
4                Этажей в доме   1.222368
5                     Парковка   0.871698
6                Площадь кухни   0.829163
7                Жилая площадь   0.770354
8            Количество комнат   0.731774
9              Высота потолков   0.722802
10                 Тип продажи   0.715445
11                        Этаж   0.540010
12                 Вид из окон   0.392359
13                    Тип дома   0.356863
14  Лифт пассажирский (кол-во)   0.253943
15  Кол-во раздельных санузлов   0.233677
16      Лифт грузовой (кол-во)   0.216285
17              Объект продажи   0.040390
18                Мусоропровод   0.012599



mse
0:	learn: 0.7027768	test: 0.6971142	best: 0.6971142 (0)	total: 113ms	remaining: 37m 32s
1000:	learn: 0.0404742	test: 0.0870261	best: 0.0870261 (1000)	total: 25.3s	remaining: 8m
2000:	learn: 0.0230327	test: 0.0832834	best: 0.0832834 (2000)	total: 49.8s	remaining: 7m 27s
3000:	learn: 0.0155536	test: 0.0823880	best: 0.0823880 (3000)	total: 1m 17s	remaining: 7m 20s
4000:	learn: 0.0115012	test: 0.0821011	best: 0.0821011 (4000)	total: 1m 41s	remaining: 6m 46s
Stopped by overfitting detector  (100 iterations wait)

bestTest = 0.08208243906
bestIteration = 4296

Shrink model to first 4297 iterations.
/Users/azatgalautdinov/PycharmProjects/price_prediction/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but PowerTransformer was fitted with feature names
  warnings.warn(
MAPE:  0.04510601551735701
                       Признак   Важность
0                        Район  34.544349
1                Общая площадь  31.025135
2                        Округ  23.570309
3          Расстояние до метро   2.949959
4                Этажей в доме   1.221482
5                     Парковка   0.870744
6                Площадь кухни   0.829528
7                Жилая площадь   0.770715
8            Количество комнат   0.731698
9              Высота потолков   0.722434
10                 Тип продажи   0.716019
11                        Этаж   0.541426
12                 Вид из окон   0.393436
13                    Тип дома   0.355964
14  Лифт пассажирский (кол-во)   0.253308
15  Кол-во раздельных санузлов   0.232854
16      Лифт грузовой (кол-во)   0.217447
17              Объект продажи   0.040504
18                Мусоропровод   0.012688

"""