import math
import numpy as np
import matplotlib.pyplot as plt
# save csv
import csv

# python2可以用file替代open



    # 写入多行用writerows



tArr = np.arange(2, 4*np.pi, 0.001, dtype='float32')
size = len(tArr)
x = np.zeros(size)
y= np.zeros(size)
phi = [2.1, 2.8, 4.1, 4.8, 6.2]
dataAll = np.zeros((5*size,4),dtype='float32')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['figure.figsize']= (8, 6)

fig = plt.figure(figsize=(12, 12), linewidth=2)

ax = fig.gca()

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

plt.title("5Spiral", fontsize=32)
plt.tick_params(width=2, labelsize=28)
index = 0
clusterID = 0
for phi_i in phi:

    for i in range(size):
        x[i] = -1*tArr[i]/8 * math.cos(tArr[i]+phi_i)
        y[i] = -1 * tArr[i] / 8 * math.sin(tArr[i] + phi_i)
        dataAll[index]=([index, x[i], y[i], clusterID ])
        index += 1

    clusterID += 1
    plt.scatter(x,y)

with open("data/5Spiral50K.csv", "w", newline ='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Index", "x", "y", "clusterID"])
    for i in range(5*size):
        writer.writerow([int(dataAll[i,0]),dataAll[i,1], dataAll[i,2], int(dataAll[i,3])])
print("data saved!")
plt.show()



