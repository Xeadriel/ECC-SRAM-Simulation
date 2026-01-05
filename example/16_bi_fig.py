import numpy as np
import matplotlib.pyplot as plt

# Open the file and read lines
with open('16_bi_data_avg.txt', 'r') as f:
    lines = f.readlines()

# Extract bet_data from the first line
bet_data_from_file = list(map(float, lines[0].strip().split('\t')))

# Extract avg_accor_data_for_bt3 from the second line
avg_accor_data_for_bt3 = list(map(float, lines[1].strip().split('\t')))

# Extract avg_values_from_null from the third line
avg_values_from_null = list(map(float, lines[2].strip().split('\t')))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(bet_data_from_file, avg_values_from_null, label='model', color='blue')
plt.plot(bet_data_from_file,  avg_accor_data_for_bt3 , label='sub model', color='red')

# Setting other plot attributes
plt.xscale('log')
plt.gca().invert_xaxis()  # Invert x-axis
plt.xlabel('Bit Error Rate')
plt.ylabel('Accuracy')
plt.title('Model vs. Sub Model Accuracy')
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
