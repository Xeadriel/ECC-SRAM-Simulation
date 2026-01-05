import numpy as np
import matplotlib.pyplot as plt

# 从4data_avg.txt读取bt_data为3的那一行的avg_accor_data
with open('4data_avg.txt', 'r') as f:
    lines = f.readlines()

bet_data_from_file = list(map(float, lines[0].strip().split('\t')))
avg_accor_data_for_bt3 = None

for line in lines[1:]:
    parts = line.strip().split('\t')
    if int(parts[0]) == 3:  # 对应于bt_data为3的那一行
        avg_accor_data_for_bt3 = list(map(float, parts[1:]))
        break

# 从4null_avg.txt读取平均值
with open('4null_avg.txt', 'r') as f:
    lines = f.readlines()

avg_values_from_null = list(map(float, lines[1].strip().split('\t')[1:]))

# 绘制折线图
plt.figure(figsize=(10, 6))

# 绘制model数据
plt.plot(bet_data_from_file, avg_accor_data_for_bt3, label='model', color='blue')

# 绘制sub model数据
plt.plot(bet_data_from_file, avg_values_from_null, label='sub model', color='red')

# 设置其他图形属性
plt.xscale('log')
plt.gca().invert_xaxis()  # x轴倒序
plt.xlabel('Bit Error Rate')
plt.ylabel('Accuracy')
plt.title('Model vs. Sub Model Accuracy')
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
