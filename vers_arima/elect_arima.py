import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time  # Для измерения времени

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Настройка стиля Seaborn для красивой визуализации
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Зафиксируем время начала всего скрипта
script_start_time = time.time()

# Загрузка данных
df = pd.read_csv(
    r"d:\data_science_24_part2\data_science_part2\household_power_consumption.csv",
    sep=';',
    low_memory=False,  # Чтобы избежать предупреждений о типах данных
    na_values=['?']  # Обозначение пропусков
)

# Предварительный просмотр данных
print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nКоличество пропусков в каждом столбце:")
print(df.isnull().sum())
print("\nСтатистическое описание данных:")
print(df.describe())

# Обработка пропусков
df.dropna(inplace=True)  # Удаляем строки с пропусками

# Преобразование столбцов в числовые типы
numeric_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
                "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
df[numeric_cols] = df[numeric_cols].astype(float)

# Преобразование столбцов Date и Time в datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)

# Отбор данных за один год
df_loc = df.loc['2006']  # Используем формат 'YYYY-MM-DD'
df_loc = df_loc.reset_index()
df_loc['ID'] = df_loc.index + 1  # Создаем последовательный ID
print("\nПервые 5 строк данных за выбранный день:")
print(df_loc.head())

# Выбор признаков и цели
data = df_loc[["Datetime", "Global_active_power"]]


# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[['Global_active_power']])

#Визуализация ряда
plt.figure(figsize=(15,8))
plt.plot(data['Global_active_power'])
plt.title('График временного ряда')
plt.xlabel('Дата')
plt.ylabel('Общая активная мощность')
plt.show()

# Проверка на стационарность с помощью ADF теста
print('Результат теста:')
df_result = adfuller(data['Global_active_power'])
df_labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
for result_value, label in zip(df_result, df_labels):
    print(label + ' : ' + str(result_value))

if df_result[1] <= 0.05:
    print("Сильные доказательства против нулевой гипотезы, ряд является стационарным.")
else:
    print("Слабые доказательства против нулевой гипотезы, ряд не является стационарным.")


# Обучение модели ARIMA
model = ARIMA(data["Global_active_power"], order=(1,1,1))

model_fit = model.fit()
print(model_fit.summary())

# Прогнозирование
forecast = model_fit.forecast(steps=1000)
print(forecast)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(data["Global_active_power"], label='Исходные данные')
plt.plot(forecast, label='Прогноз')
plt.title('Прогнозирование временного ряда')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.show()

# Оценка модели
data_train = data["Global_active_power"].iloc[:-10]
data_test = data["Global_active_power"].iloc[-10:]
model = ARIMA(data["Global_active_power"], order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
mse = mean_squared_error(data_test, forecast)
print(f'Mean Squared Error: {mse}')


# Закончим скрипт и выведем общее время выполнения
script_end_time = time.time()
script_duration = script_end_time - script_start_time
print(f"\nОбщее время выполнения скрипта: {script_duration:.2f} секунд")
