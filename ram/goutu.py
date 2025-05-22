import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=["SimHei"] #用来正常显示中文标签
import numpy as np

# 定义横轴和纵轴刻度
x_ticks = [0, 25, 50, 75]


# 第一张图的数据 (λ 取值为 1,10, 25, 100, 200, 600)
x1 = np.array([0, 25, 50, 75])
y1 = {
    10: [69.19, 70.42, 72.99, 73.96],
    25: [69.32, 70.44, 73.50, 74.10],
    100: [68.46, 70.10, 72.99, 74.12],
    200: [69.12, 70.08, 72.69, 73.11],
    500: [68.09, 69.85, 72.56, 72.98]
}
# 第一张图的纵轴刻度
y1_ticks = [69, 70, 71, 72, 73, 74]

# 第二张图的数据 (PN 取值为 0, 1, 10, 20, 40)
x2 = np.array([0, 25, 50, 75])
y2 = {
    0: [68.10, 69.32, 72.66, 73.06],
    1: [69.21, 70.26, 73.25, 74.02],
    10: [69.33, 70.39, 73.42, 74.15],
    20: [69.32, 70.44, 73.50, 74.10],
    60: [69.18, 70.31, 73.36, 73.85]
}
# 第二张图的纵轴刻度
y2_ticks = [68, 70, 72, 74, 76]

# 设置绘图区域大小
plt.figure(figsize=(12, 5))

# 第一张图
plt.subplot(1, 2, 2)
for lambda_val, y_vals in y1.items():
    plt.plot(x1, y_vals, marker='s', label=f'λ={lambda_val}')
plt.xlabel('开放词汇比例',fontsize=18)
plt.ylabel(' mAP (%)',fontsize=18)
plt.xticks(x_ticks)
plt.yticks(y1_ticks)
plt.legend()
plt.grid(True)

# 第二张图
plt.subplot(1, 2, 1)
for pn_val, y_vals in y2.items():
    plt.plot(x2, y_vals, marker='s', label=f'PN={pn_val}')
plt.xlabel('开放词汇比例',fontsize=18)
plt.ylabel(' mAP (%)',fontsize=18)
plt.xticks(x_ticks)
plt.yticks(y2_ticks)
plt.legend()
plt.grid(True)

# 调整两个图表之间的距离
plt.subplots_adjust(wspace=0.4)  # 通过wspace参数增加图表之间的间距

# 调整布局并显示图像
# plt.tight_layout()
plt.show()
