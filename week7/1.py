import numpy as np
import pandas as pd
import sys
import concurrent.futures
from multiprocessing import Pool
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression


def show_nan(data):
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    np.set_printoptions(threshold=sys.maxsize)
    print('columns count values')
    print(data.count())
    cols_nan = data.columns[data.isna().any()]
    print('columns nan', cols_nan)


def drop_match_results(data):
    return data.drop(['duration', 'radiant_win',
                      'tower_status_radiant', 'tower_status_dire',
                      'barracks_status_radiant', 'barracks_status_dire'], axis=1)

# Кроссвалидация с градиентным бустингом
def crossval_gradient(X, y, kf):
    best_score = 0
    for n_est in range(10, 41, 10):
        for learning_rate in [1.5]:
            classifier = GradientBoostingClassifier(n_estimators=n_est, verbose=True,
                                                    random_state=42,
                                                    max_features=10, subsample=1.0, max_depth=3,
                                                    learning_rate=learning_rate)
            cv_stat = cross_validate(estimator=classifier, X=X, y=y, cv=kf, scoring='roc_auc', n_jobs=10)
            print(' --- TIME: ', np.average(cv_stat['fit_time']))
            new_score = np.average(cv_stat['test_score'])
            if new_score > best_score:
                best_score = new_score
                best_n = n_est
                print(' --- SCORE: ', best_score, best_n)


def get_hero_bag_column_names(n):
    columns = []
    for i in np.arange(1, n+1):
        columns.append('hero_%d' % i)
    return columns


def get_hero_col_names():
    return ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']


def get_hero_count(data):
    heroes_unique_ids = np.unique(data.get(get_hero_col_names()).values)  # 108
    hero_count = heroes_unique_ids.max()  # 112
    return hero_count


# Замена мешком слов
def replace_with_bag(args):
    proc_data = args[0]
    hero_count = args[1]
    match_id_min = proc_data.index.min()
    match_id_max = proc_data.index.max()
    id_range = np.arange(match_id_min, match_id_max + 1, 1)
    X_pick = pd.DataFrame(data=np.zeros((id_range.size, hero_count)),
                          columns=get_hero_bag_column_names(hero_count),
                          index=id_range,
                          dtype=int)
    for match_id, match_data in proc_data.iterrows():
        for p in np.arange(1, 6):
            radiant_hero_id = int(match_data.loc['r%d_hero' % p])
            dire_hero_id = int(match_data.loc['d%d_hero' % p])
            X_pick.at[match_id, 'hero_%d' % radiant_hero_id] = 1
            X_pick.at[match_id, 'hero_%d' % dire_hero_id] = -1
    X_pick = X_pick.loc[(X_pick!=0).any(axis=1)]
    return X_pick


def log_reg(X, y, kf):
    best_score = 0
    for c in np.arange(0.0045, 0.0060, 0.0001):
        classifier = LogisticRegression(random_state=42, C=c)
        cv_stat = cross_validate(estimator=classifier, X=X, y=y, cv=kf, scoring='roc_auc')
        print(' --- TIME: ', np.average(cv_stat['fit_time']))
        new_score = np.average(cv_stat['test_score'])
        if new_score > best_score:
            best_score = new_score
            best_c = c
            print(' --- SCORE: ', best_score, best_c)


def add_x_pick_table(data):
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # Раскидываем по процессам
        jobs = []
        count = get_hero_count(data)
        for proc_data in np.array_split(data, 10):
            proc_data = proc_data.loc[:, get_hero_col_names()]
            jobs.append(executor.submit(replace_with_bag, (proc_data, count)))
        # Собираем результаты
        results = None
        for job in jobs:
            if results is None:
                results = job.result()
            else:
                results = pd.concat([results, job.result()])
        return pd.concat([data, results], axis=1, sort=False)


def main():

    # Подход 1: градиентный бустинг "в лоб"

    data = pd.read_csv('features.csv', index_col='match_id')
    show_nan(data)
    data = data.fillna(value=10000)
    X = drop_match_results(data)
    y = data['radiant_win'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    crossval_gradient(X, y, kf)

    # # # Отчет
    # 1. столбцы с пропусками: 'first_blood_time', 'first_blood_team', 'first_blood_player1',
    #        'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time',
    #        'radiant_flying_courier_time', 'radiant_first_ward_time',
    #        'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time',
    #        'dire_first_ward_time'
    #    причины пропусков - статистически редкие события
    #    (до 5 минуты не сделали фб, не купили курьера, не ставили вардов, не прожимали бутылку)
    # 2. 'radiant_win'
    # 3. 29 секунд, если строить в лоб; 3.3 секунды, если ограничить построение 10 признаками на процесс
    # 4. в лоб достаточно 30 деревьев для качества 0.6833, дальше растет медленно, пока не начнет переобучаться.
    # Eсли ограничивать выборку или набор признаков, то качество падает на 1-2% и может иметь смысл дальнейшее построение;
    # Ограничение выборки ухудшает качество в пределах 1-4% - модель чувствительна к размеру обучающей выборки;
    # Глубина деревьев влияет слабо, т.к. зависимости простые;
    # # #

    # Подход 2: логистическая регрессия

    # На всех столбцах
    X = drop_match_results(data)
    X = scale(X)
    log_reg(X, y, kf)

    # Без категориальных
    X = drop_match_results(data)
    X = X.drop(['lobby_type'] + get_hero_col_names(), axis=1)
    X = scale(X)
    log_reg(X, y, kf)

    # С мешком слов
    print('hero count: %d' % get_hero_count(data))
    X = drop_match_results(data)
    X = add_x_pick_table(X)
    X = X.drop(['lobby_type'] + get_hero_col_names(), axis=1)
    X = scale(X)
    log_reg(X, y, kf)

    # Обучаем классификатор по лучшему варианту
    classifier = LogisticRegression(random_state=42, C=0.0058)
    classifier.fit(X, y)
    # Загружаем и предобрабатываем тестовые данных
    data_test = pd.read_csv('features_test.csv', index_col='match_id')
    data_test = data_test.fillna(value=10000)
    X_test = add_x_pick_table(data_test)
    X_test = X_test.drop(['lobby_type'] + get_hero_col_names(), axis=1)
    X_test = scale(X_test)
    # Находим лучший прогноз
    proba = classifier.predict_proba(X_test)
    print(proba.min(), proba.max())
    proba_frame = pd.DataFrame(data=proba[:, 1],
                               columns=['radiant_win'],
                               index=data_test.index)
    proba_frame.index.name = 'match_id'
    proba_frame.to_csv(path_or_buf="test_proba.csv")

    # # # Отчет
    # 1. Среднее качетсво у логистической регрессии над всеми исходными признаками = 0.72
    # Немного лучше чем у бустинга, т.к. данных мало, вероятно, они разреженные и с аномалиями.
    # Логистическая регрессия работает примерно в 2 раза быстрее, с учетом параллезизации, без учета предобработки.
    # 2. Удаление категориальных признаков почти не влияет на качество, без предобработки это шум с нулевым весом.
    # 3. В рассматриваемой версии игры 112 героев, в данных найдено 108 различных идентификаторов.
    # 4. При добавлении "мешка слов" по героям качество улучшилось до 0.75.
    # Без кодирования герои не учитываются в прогнозе, но влияют на результат матча.
    # 5. На тестовой выборке получилось максимум 100% и 0% на победу/поражение, округляя до 2 знаков.


# Это нужно для многопоточности
if __name__ == '__main__':
    main()

