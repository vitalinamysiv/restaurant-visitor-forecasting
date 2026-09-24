# Restaurant Visitor Forecasting

## Описание проекта

Проект направлен на прогнозирование количества посетителей ресторанных точек на следующие 7 дней.

В качестве целевой переменной используется:
- guests — количество посетителей в день.

Дополнительный бизнес-показатель:
- revenue — дневная выручка.

---

## Датасет

Использован открытый датасет Rossmann Store Sales.

Исходные данные содержат:
- дату;
- идентификатор точки;
- продажи;
- количество клиентов.

Для задачи прогнозирования:
- Customers → guests;
- Sales → revenue.

---

## Структура проекта


restaurant-visitor-forecasting/

├── data/
│ ├── raw/
│ └── processed/

├── models/
│ └── catboost_model.pkl

├── notebooks/
│ ├── 00_prepare_data.ipynb
│ ├── 01_eda.ipynb
│ └── 02_modeling.ipynb

├── src/
│ ├── create_dataset.py
│ ├── features.py
│ ├── validation.py
│ └── model.py

├── predict.py
├── requirements.txt
└── README.md

---

## Подготовка данных

Исходный датасет:
Rossmann Store Sales

Для подготовки данных:

python src/create_dataset.py

Скрипт создаёт:

data/processed/restaurant_daily.csv


---

## Feature engineering

Использованы:

- день недели;
- месяц;
- выходной день;
- lag признаки:
  - lag_1;
  - lag_7;
  - lag_14;
- rolling statistics:
  - rolling_mean_7;
  - rolling_mean_28.

---

## Валидация

Использовано временное разделение:

Train:
2013-01-29 — 2015-06-19

Validation:
2015-06-20 — 2015-07-31

Перемешивание данных не использовалось.

---

## Результаты моделей

|Модель|MAE|MAPE|
|-|-|-|
|Baseline lag_7|163.4|24.9%|
|Ridge|69.7|10.3%|
|CatBoost|49.8|7.1%|

---

## Запуск проекта

Установить зависимости:

pip install -r requirements.txt


Подготовить данные:

python src/create_dataset.py


Запустить прогноз:

python predict.py --date 2015-07-01 --restaurant 1
