import matplotlib.pyplot as plt
from numpy import mean

plt.rcParams['font.family'] = 'Microsoft YaHei'
location = ['Siamban','本文模型','Siamese_CAR','Siamese_Concat','Siamese_Mask','Siamese_RBO']
avg_kill = [0.441,0.452,0.395,0.364,0.446,0.475]
plt.figure(figsize=(10, 10), dpi=100)
plt.bar(location, avg_kill, width=0.05, color=['b', 'y', 'c', 'r', 'g','#b03060'])
plt.xticks(fontsize=10)
plt.yticks(range(0, 2, 1), fontsize=14)
for a, b in zip(range(6), avg_kill):
    plt.text(a, b+0.1, '%.03f' % b, ha='center', va='bottom', fontsize=14)
plt.grid(linestyle="--", alpha=0.1)

plt.xlabel("不同损失方案", fontsize=16)
plt.ylabel("平均期望重叠EAO", fontsize=16)
plt.savefig("折线图.jpg")
plt.show()
