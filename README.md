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
Весь массив данных очень большой, анализ осуществлялся за дату 17/12/2006  
Была применена модель Gradient Boosting Regressor  
Получены метрики  
MAE: 0.03281071087734397  
MSE: 0.002049100825760913  
R²: 0.9987154855018333  
Однако график выглядит грустно  

