import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv("d:\\data_science_24_part2\data_science_part2\household_power_consumption.csv", sep=';')
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
# многомерный набор данных временных рядов, который описывает потребление электроэнергии одним домохозяйством за четыре года
# с декабря 2006 года по ноябрь 2010 года. Наблюдения собирались каждую минуту
# global_active_power: общая активная мощность, потребляемая домохозяйством
# global_reactive_power: общая реактивная мощность, потребляемая домохозяйством
# global_intensity: средняя сила тока (в амперах).
# sub_metering_1: Активная энергия для кухни (ватт-часы активной энергии).
# sub_metering_2: Активная энергия для стирки (ватт-часы активной энергии).
# sub_metering_3: Активная энергия для систем климат-контроля (ватт-часы активной энергии).
# Отберем данные за один день

df_loc=df[df["Date"] == '17/12/2006']
df_loc['ID'] = df_loc.index + 1
print(df_loc.head())
data=df_loc[["ID","Global_reactive_power","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]]
target = df_loc["Global_active_power"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Создание и обучение модели Gradient Boosting Regressor
model = GradientBoostingRegressor() # параметры
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'*****************')
print(f'Метрики качества:')
print(f'*****************')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Визуализация предсказанных и фактических значений
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.show()