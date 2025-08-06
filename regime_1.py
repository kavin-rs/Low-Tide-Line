'''
该代码为针对单条剖面线的点进行突变点检测和早期预警信号（自相关和方差）提取
输入距离点数据，指定剖面线位置，输出突变点检测和早期预警信号序列，也可以作图
'''

from regimeshifts import regime_shifts as rs
from regimeshifts import ews
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# 读取CSV文件
file_path = './01_dataset/intersects_2024.csv'
data = pd.read_csv(file_path)

#结果保存路径
result_path = './02_result/'

# 确保ShorelineI是日期格式
data['ShorelineI'] = pd.to_datetime(data['ShorelineI'], format='%d/%m/%Y')

# 获取唯一的TransectID值
transect_id = 482
# 438 强突变  405  弱突变  175  持续性侵蚀  528 弱突变侵蚀侵蚀

# 筛选当前TransectID的数据
transect_data = data[data['TransectID'] == transect_id]

# 按照ShorelineI排序
transect_data_sorted = transect_data.sort_values(by='ShorelineI')

# 提取排序后的Distance列
distances = transect_data_sorted['Distance']

#将数据归一化到[0,1]
X_min = distances.min(axis=0)
X_max = distances.max(axis=0)
distances = (distances - X_min) / (X_max - X_min)

# 已知的数据点
x = np.arange(1, len(distances)+1, 1)  # 假设数据点是等间距的，这里用数据的索引作为x坐标
distances = np.array(distances)

# 创建插值函数，选择插值方法，例如线性插值
linear_interp = interp1d(x, distances, kind='linear')

# 定义新的插值点
x_line = np.linspace(x.min(), x.max(), 400)  # 生成300个介于最小和最大x之间的新点

# 使用插值函数计算新的y值
distances_line = linear_interp(x_line)


# 线性拟合Distance和日期
# 将日期转换为数值型（例如：距离1970年的天数）
# dates = transect_data_sorted['ShorelineI'].map(pd.Timestamp.toordinal)

# 创建一个新数组，每隔10个位置放入一个归一化的值
# distances_nan = np.full(390, np.nan)  # 创建一个长度为390，初始值为NaN的数组
# step_size = 10  # 每隔10个位置插入一个归一化的值
# distances_nan[::step_size] = distances

#对插值后的结果进行检测
distances_line = rs.Regime_shift(distances_line)
# detection_index = distances_line.as_detect()

series = ews.Ews(distances_line)
## The Ews class returns an extended Dataframe object, if we provided a series, it sets 0 for the column name.
series = series.rename(columns={0:'Sample series'})

# trend = series.gaussian_det(bW=10).trend
# residuals = series.gaussian_det(bW=50).res['Sample series']

wL = 30 ## Window length specified in number of points in the series
bW = 20
# ar1 = series.ar1(detrend=True,bW=bW,wL=wL)['Sample series'] ### Computing lag-1 autocorrelation using the ar1() method
var = series.var(detrend=True,bW=bW,wL=wL)['Sample series'] ## Computing variance

# 创建一个包含所有数据的 DataFrame
origin_to_save = pd.DataFrame({
    'X': x,
    'Distances': distances,
})

# 创建一个包含所有数据的 DataFrame
data_to_save = pd.DataFrame({
    'X': x_line,
    'Normalized_Distances': distances_line,
    # 'detection_index': detection_index,
    # 'residuals': residuals,
    # 'ar1': ar1,
    'var': var
})
abs_max_value = np.max(np.abs(var)) #找到绝对值最大值
abs_max_index = np.argmax(np.abs(var))  # 找到绝对值最大的元素的索引
# print(detection_index.max())
# print(np.argmax(detection_index))
print(abs_max_value)
print(abs_max_index)
fig, ax = plt.subplots()
distances_line.plot(ax=ax)
ax.set_xlabel('Time',fontsize=12)
ax.set_ylabel('System state',fontsize=12)

# fig, ax = plt.subplots()
# residuals.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('Detection Index',fontsize=12)
# ax.set_ylim(-1, 1)

# fig, ax = plt.subplots()
# ar1.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('Detection Index',fontsize=12)
# ax.set_ylim(-1, 1)

fig, ax = plt.subplots()
var.plot(ax=ax)
ax.set_xlabel('Time',fontsize=12)
ax.set_ylabel('Detection Index',fontsize=12)
# ax.set_ylim(0, 0.002)

# fig, ax = plt.subplots()
# var_trend.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('Detection Index',fontsize=12)

# fig, ax = plt.subplots()
# detection_index.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('Detection Index',fontsize=12)
# ax.set_ylim(-1, 1)

# fig, ax = plt.subplots()
# diff_series.plot(ax=ax)
# ax.set_xlabel('Time',fontsize=12)
# ax.set_ylabel('Detection Index',fontsize=12)
# # ax.set_ylim(-1, 1)
plt.show()

# 保存到 CSV 文件
res_origin = os.path.join(result_path, str(transect_id)+'_origin_res.csv')
res_detect = os.path.join(result_path, str(transect_id)+'_detect_res.csv')
# res_ews = os.path.join(result_path, str(transect_id)+'_ews_res.csv')

origin_to_save.to_csv(res_origin, index=False)
data_to_save.to_csv(res_detect, index=False)

print(f"数据已保存到 {result_path}")





