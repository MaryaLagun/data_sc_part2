Домашнее задание 1.  
Каждое из этих заданий предоставляет возможность применить различные методы машинного обучения, которые мы рассмотрели на лекции, оценить их производительность и провести анализ результатов.  
Решите задачу регрессии на наборе данных о расходе электроэнергии в домах и предскажите будущий расход энергии.  
Для решения задачи был выбраннНабор данных о потреблении электроэнергии домохозяйствами - это многомерный набор данных временных рядов, который описывает потребление электроэнергии одним домохозяйством  
за четыре года с декабря 2006 года по ноябрь 2010 года. Наблюдения собирались каждую минуту  

  Набор данных включает показатели по конкретному использованию энергии.  
В документации указаны следующие представляющие интерес переменные:  

•	global_active_power: общая активная мощность, потребляемая домохозяйством (киловатты)  
•	global_reactive_power: общая реактивная мощность, потребляемая домохозяйством (киловатты). напряжение: Среднее напряжение (вольт)  
•	global_intensity: средняя сила тока (в амперах)  
•	sub_metering_1: Активная энергия для кухни (ватт-часы активной энергии)  
•	sub_metering_2: Активная энергия для стирки (ватт-часы активной энергии)  
•	sub_metering_3: Активная энергия для систем климат-контроля (ватт-часы активной энергии)  
Это временной ряд и прогнозировать его наверно надо с помощью ARIMA  
Попробуем, что получиться с помощью регрессии  
Весь массив данных очень большой, анализ осуществлялся за 2006 год  
Была применена модель Gradient Boosting Regressor  
Получены метрики  
MAE: 0.0340  
MSE: 0.0027  
R²: 0.9986  
Время обучения модели: 1.68 секунд  
Общее время выполнения скрипта: 270.95 секунд   
![result_fact_2006](https://github.com/user-attachments/assets/168f4c9c-3490-446e-93ee-35f73cd6b7ab)

  

![result_2006](https://github.com/user-attachments/assets/659e7124-34ba-4bb1-b149-21fcc185cd3d)  


Анализ остатов имеет вид   

![ost_2006](https://github.com/user-attachments/assets/011fac2b-c029-4991-b28a-abc6310ade3e)  

Ошибки предсказания имеют нормальное распределение   

![rasp_2006](https://github.com/user-attachments/assets/0b47ab23-f782-4e5d-a1a9-e47ccffd87e0)  




