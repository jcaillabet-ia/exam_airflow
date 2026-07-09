from airflow import DAG
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import ast
from datetime import datetime, timedelta
from joblib import dump
import json
import os
import requests
import pandas as pd

def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        # creating target
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

        # creating features
        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    return features, target

def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score

def transform_data_into_csv(n_files=None, filename='data.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []

    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pression': data_city['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)

def train_and_save_model(model, X, y, path_to_model='/app/model.pckl'):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)

@task
def task1():
    cities = Variable.get(key="cities")
    cities = ast.literal_eval(cities)
    api_key = Variable.get(key="openweathermap_api_key")
    now = datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M")
    cities_api = []
    for city in cities:
        print("https://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=" + api_key)
        response = requests.get("https://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=" + api_key)
        cities_api.append(response.json())
    with open('/app/raw_files/' + date + '.json', 'w') as f:
        json.dump(cities_api, f)

@task
def task2():
    transform_data_into_csv(20)

@task
def task3():
    transform_data_into_csv(filename='fulldata.csv')

@task
def task4_1(ti=None):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_lr = compute_model_score(LinearRegression(), X, y)

    ti.xcom_push(
        key="score_lr",
        value=score_lr
    )

@task
def task4_2(ti=None):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_dtr = compute_model_score(DecisionTreeRegressor(), X, y)

    ti.xcom_push(
        key="score_dtr",
        value=score_dtr
    )

@task
def task4_3(ti=None):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_rfr = compute_model_score(RandomForestRegressor(), X, y)

    ti.xcom_push(
        key="score_rfr",
        value=score_rfr
    )

@task
def task5(ti=None):
    score_lr = ti.xcom_pull(
        key="score_lr",
        task_ids='task4_1'
    )
    score_dtr = ti.xcom_pull(
        key="score_dtr",
        task_ids='task4_2'
    )
    score_rfr = ti.xcom_pull(
        key="score_rfr",
        task_ids='task4_3'
    )   
    
    max_score = max(score_lr, score_dtr, score_rfr)

    X, y = prepare_data('/app/clean_data/fulldata.csv')

    if score_lr == max_score:
        train_and_save_model(
            LinearRegression(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )
    elif score_dtr == max_score:
        train_and_save_model(
            DecisionTreeRegressor(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )
    elif score_rfr == max_score:
        train_and_save_model(
            RandomForestRegressor(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )

@dag(
    dag_id='exam_airflow_v2',
    tags=['exam'],
    schedule_interval = "* * * * *",
    start_date=days_ago(0),
    catchup=False
)
def my_dag():
    t1 = task1()
    t2 = task2()
    t3 = task3()
    t4_1 = task4_1()
    t4_2 = task4_2()
    t4_3 = task4_3()
    t5 = task5()

    t1 >> [t2,t3]
    t3 >> [t4_1, t4_2, t4_3]
    [t4_1, t4_2, t4_3] >> t5

my_dag = my_dag()
