import numpy as np
import matplotlib.pyplot as plt


bet_data=(1,0.7943282347242815,0.6309573444801932,0.5011872336272722,0.3981071705534972,0.31622776601683794,0.251188643150958,0.19952623149688797,0.15848931924611134,0.12589254117941673,0.1,0.07943282347242814,0.0630957344480193,0.05011872336272722,0.03981071705534971,0.0316227766016838,0.025118864315095794,0.019952623149688786,0.015848931924611134,0.012589254117941668,0.01, 0.007943282347242814, 0.00630957344480193, 0.005011872336272719, 0.003981071705534969, 0.003162277660168376,
    0.002511886431509577, 0.0019952623149688768, 0.0015848931924611108, 0.0012589254117941649, 0.0009999999999999979,
    0.0007943282347242797, 0.0006309573444801917, 0.0005011872336272709, 0.0003981071705534961, 0.00031622776601683696,
    0.0002511886431509572, 0.00019952623149688728, 0.00015848931924611077, 0.00012589254117941623,
    9.999999999999958e-05, 7.94328234724279e-05, 6.309573444801917e-05, 5.011872336272715e-05, 3.9810717055349695e-05,
    3.9810717055349695e-05)
    #  bet_data=(1,0.6309573444801932)
   # bet_data= bet_data1[::-1]
bt_data = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
accor_data = np.zeros((len(bet_data), len(bt_data)))
# 读取数据文件
with open('16data_avg.txt', 'r') as f:
    lines = f.readlines()

# 提取bet_data
bet_data_from_file = list(map(float, lines[0].strip().split('\t')))

# 提取accor_data
accor_data_from_file = np.zeros((len(bet_data_from_file), len(bt_data)))
for idx, line in enumerate(lines[1:]):
    parts = line.strip().split('\t')
    accor_data_from_file[:, idx] = list(map(float, parts[1:]))

# 绘制折线图
for i, bt in enumerate(bt_data):
    plt.plot(bet_data_from_file, accor_data_from_file[:, i], label=f'Bit[{bt}]')

plt.xscale('log')
plt.gca().invert_xaxis()  # 倒序x轴

plt.xlim(0.0316227766016838, 0.003162277660168376)
plt.ylim(0.4, 1)

plt.xlabel('Bit Error Rate')
plt.ylabel('Accurate Recognition')
plt.legend(loc='lower right', fontsize='x-small')
plt.title('Bit Error Rate and Accurate Recognition from File')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
