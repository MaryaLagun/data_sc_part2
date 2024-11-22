pipimport torch

# Загрузка данных
FILE = "data.pth"
data = torch.load(FILE)

# Печать ключей
print(data.keys())