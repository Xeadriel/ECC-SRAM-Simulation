import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设定字体和大小
matplotlib.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# 读取数据
bet_data = (0.01, 0.007943282347242814, 0.00630957344480193, 0.005011872336272719, 0.003981071705534969, 0.003162277660168376,
    0.002511886431509577, 0.0019952623149688768, 0.0015848931924611108, 0.0012589254117941649, 0.0009999999999999979,
    0.0007943282347242797, 0.0006309573444801917, 0.0005011872336272709, 0.0003981071705534961, 0.00031622776601683696,
    0.0002511886431509572, 0.00019952623149688728, 0.00015848931924611077, 0.00012589254117941623,
    9.999999999999958e-05, 7.94328234724279e-05)
bt_data = (0, 1, 2, 3, 4, 5, 6, 7)
accor_data = np.zeros((len(bet_data), len(bt_data)))

# 读取数据文件
with open('8bit.txt', 'r') as f:
    lines = f.readlines()

# 提取bet_data
bet_data_from_file = list(map(float, lines[0].strip().split('\t')))

# 提取accor_data
accor_data_from_file = np.zeros((len(bet_data_from_file), len(bt_data)))
for idx, line in enumerate(lines[1:]):
    parts = line.strip().split('\t')
    accor_data_from_file[:, idx] = list(map(float, parts[0:]))

# 绘制折线图
for i, bt in enumerate(bt_data):
    plt.plot(bet_data_from_file, accor_data_from_file[:, i], label=f'Bit[{bt}]')

# 设置坐标轴为对数刻度
plt.xscale('log')
plt.gca().invert_xaxis()

# 设置坐标轴范围和标签
plt.xlim(0.01, 7.94328234724279e-05)
plt.ylim(0.4, 1)
plt.xlabel('Bit Error Rate', fontsize=14)
plt.ylabel('Accurate Recognition Rate',fontsize=14)

# 调整图例
plt.legend(loc='lower right',fontsize='small')

# 添加标题和网格线
# plt.title('Bit Error Rate and Accurate Recognition from File')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 调整图表尺寸以适应论文格式
plt.gcf().set_size_inches(8, 6)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
# 显示图表
plt.show()
