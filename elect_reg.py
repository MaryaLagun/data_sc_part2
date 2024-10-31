import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time  # Для измерения времени

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
print("\nПервые 5 строк данных за выбранный год:")
print(df.head())

# Выбор признаков и цели
data = df_loc[["ID", "Global_reactive_power", "Global_intensity",
              "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]
target = df_loc["Global_active_power"]

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.2, random_state=42
)

# Создание и обучение модели Gradient Boosting Regressor
model_start_time = time.time()
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
model_end_time = time.time()
model_training_time = model_end_time - model_start_time
print(f"\nВремя обучения модели: {model_training_time:.2f} секунд")

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\n*****************')
print(f'Метрики качества:')
print(f'*****************')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')

# Создадим DataFrame для удобства визуализации
results = pd.DataFrame({
    'ID': df_loc.loc[y_test.index, 'ID'].values,
    'Actual': y_test.values,
    'Predicted': y_pred
}).sort_values('ID').reset_index(drop=True)

# График 1: Scatter plot с линией y=x
plt.figure(figsize=(8, 8))
sns.scatterplot(x='Actual', y='Predicted', data=results, alpha=0.6)
plt.plot([results['Actual'].min(), results['Actual'].max()],
         [results['Actual'].min(), results['Actual'].max()],
         color='red', linestyle='--', label='y = x')
plt.xlabel('Фактические значения (Global_active_power)')
plt.ylabel('Предсказанные значения (Global_active_power)')
plt.title('Фактические vs Предсказанные значения')
plt.legend()
plt.show()

# График 2: Линейный график для последовательности наблюдений
plt.figure(figsize=(14, 6))
plt.plot(results['ID'], results['Actual'], label='Фактические значения', color='blue')
plt.plot(results['ID'], results['Predicted'], label='Предсказанные значения', color='red', alpha=0.7)
plt.xlabel('ID (Порядковый номер наблюдения)')
plt.ylabel('Global_active_power')
plt.title('Фактические и Предсказанные значения за один год')
plt.legend()
plt.show()

# График 3: Распределение ошибок предсказания
results['Error'] = results['Actual'] - results['Predicted']
plt.figure(figsize=(10,6))
sns.histplot(results['Error'], bins=50, kde=True, color='purple')
plt.xlabel('Ошибка (Фактические - Предсказанные)')
plt.title('Распределение ошибок предсказания')
plt.show()

# График 4: Боксплот ошибок
plt.figure(figsize=(8,6))
sns.boxplot(x=results['Error'], color='lightgreen')
plt.xlabel('Ошибка (Фактические - Предсказанные)')
plt.title('Боксплот ошибок предсказания')
plt.show()

# График 5: Корреляционная матрица между фактическими и предсказанными значениями
plt.figure(figsize=(6,4))
corr = results[['Actual', 'Predicted']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Корреляция между фактическими и предсказанными значениями')
plt.show()

# Дополнительная визуализация: Важности признаков
feature_names = ["ID", "Global_reactive_power", "Global_intensity",
                 "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
feature_importances = pd.Series(model.feature_importances_, index=feature_names)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.xlabel('Важность признака')
plt.ylabel('Признак')
plt.title('Важности признаков в модели Gradient Boosting Regressor')
plt.show()

# График 6: Анализ остатков
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_pred, y=results['Error'], alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки (Фактические - Предсказанные)')
plt.title('Анализ остатков')
plt.show()

# Закончим скрипт и выведем общее время выполнения
script_end_time = time.time()
script_duration = script_end_time - script_start_time
print(f"\nОбщее время выполнения скрипта: {script_duration:.2f} секунд")
