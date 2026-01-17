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
        learning_rate=0.05,
        depth=10,
        loss_function='MAE',  # 'Huber:delta=1.0',
        # eval_metric='MAPE',
        # custom_metric=mape,
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

    model.save_model('price_prediction.cbm')

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
0:	learn: 0.9611843	test: 0.9574410	best: 0.9574410 (0)	total: 89.4ms	remaining: 29m 47s
1000:	learn: 0.0925948	test: 0.1510753	best: 0.1510715 (998)	total: 35.1s	remaining: 11m 6s
2000:	learn: 0.0647928	test: 0.1441555	best: 0.1441493 (1999)	total: 1m 13s	remaining: 10m 58s
3000:	learn: 0.0510592	test: 0.1419336	best: 0.1419336 (2999)	total: 1m 48s	remaining: 10m 13s
4000:	learn: 0.0423678	test: 0.1408791	best: 0.1408765 (3996)	total: 2m 23s	remaining: 9m 35s
5000:	learn: 0.0361971	test: 0.1403368	best: 0.1403367 (4999)	total: 3m	remaining: 9m
6000:	learn: 0.0314793	test: 0.1400304	best: 0.1400303 (5999)	total: 3m 35s	remaining: 8m 23s
7000:	learn: 0.0280239	test: 0.1398106	best: 0.1398106 (7000)	total: 4m 11s	remaining: 7m 47s
8000:	learn: 0.0250604	test: 0.1397189	best: 0.1397153 (7947)	total: 4m 48s	remaining: 7m 12s
9000:	learn: 0.0225972	test: 0.1396240	best: 0.1396166 (8904)	total: 5m 26s	remaining: 6m 38s
Stopped by overfitting detector  (200 iterations wait)

bestTest = 0.139577544
bestIteration = 9603

Shrink model to first 9604 iterations.
/Users/azatgalautdinov/PycharmProjects/price_prediction/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but PowerTransformer was fitted with feature names
  warnings.warn(
MAPE:  0.045554759900980964
                       Признак   Важность
0                        Округ  26.520317
1                Общая площадь  26.049735
2                        Район   9.846700
3          Расстояние до метро   8.397434
4                     Парковка   5.391764
5                Этажей в доме   4.171155
6                  Тип продажи   3.443368
7              Высота потолков   2.594383
8                Площадь кухни   2.511524
9                         Этаж   2.236118
10                 Вид из окон   1.840830
11                    Тип дома   1.557816
12               Жилая площадь   1.476015
13      Лифт грузовой (кол-во)   1.349974
14           Количество комнат   0.985569
15  Кол-во раздельных санузлов   0.905528
16  Лифт пассажирский (кол-во)   0.663570
17              Объект продажи   0.055959
18                Мусоропровод   0.002240

"""