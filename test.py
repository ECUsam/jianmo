import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib.patches import Circle
from sklearn.cluster import DBSCAN
from collections import Counter
import warnings

from tools import find_nearest_category

warnings.filterwarnings('ignore')


def filter_peak_hours(data):
    data['hour'] = pd.to_datetime(data['starttime']).dt.hour
    morning_peak = data[(data['hour'] >= 7) & (data['hour'] <= 9)]
    evening_peak = data[(data['hour'] >= 17) & (data['hour'] <= 19)]
    return morning_peak, evening_peak


# 给定横坐标和纵坐标范围
x_min = 116.17561340332
x_max = 116.649398803711
y_min = 39.7492218017578
y_max = 40.0952911376953

# 将范围均匀划分为48个区域
num_rows = 6  # 行数
num_cols = 8  # 列数

# 计算每个区域的宽度和高度
x_step = (x_max - x_min) / num_cols
y_step = (y_max - y_min) / num_rows

# 生成网格索引和标签的映射
label_map = {}
label_counter = 1

for row in range(num_rows):
    for col in range(num_cols):
        x_start = x_min + col * x_step
        x_end = x_start + x_step
        y_start = y_min + row * y_step
        y_end = y_start + y_step

        label_map[(row, col)] = label_counter
        label_counter += 1

# 假设你有经纬度数据
shuju = pd.read_csv('output_test.csv')
shuju, shuju_ = filter_peak_hours(shuju)

# 去掉超出范围的数据点
shuju = shuju[(shuju['geohashed_start_loc_Longitude'] >= x_min) & (shuju['geohashed_start_loc_Longitude'] <= x_max) &
              (shuju['geohashed_start_loc_Latitude'] >= y_min) & (shuju['geohashed_start_loc_Latitude'] <= y_max)]
data = pd.DataFrame({
    'longitude': shuju['geohashed_start_loc_Longitude'],  # 你的经度数据列
    'latitude': shuju['geohashed_start_loc_Latitude'],  # 你的纬度数据列
    'orderid': shuju['orderid']
})

# 计算每个数据点的行列索引，并映射为标签
data['row'] = ((data['latitude'] - y_min) / y_step).astype(int)
data['col'] = ((data['longitude'] - x_min) / x_step).astype(int)
data['label'] = data.apply(lambda row: label_map.get((row['row'], row['col'])), axis=1)

labels_to_remove = [19, 20, 21, 27, 28, 29]
removed_data = data[data['label'].isin(labels_to_remove)]
data = data[~data['label'].isin(labels_to_remove)]

# 进行 KMeans 聚类
num_clusters = 40
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
data['cluster'] = kmeans.fit_predict(data[['longitude', 'latitude']])

# 添加之前删除的数据
data = pd.concat([data, removed_data], ignore_index=True)

# 给每个类别添加新标签
cluster_labels = {cluster_id: new_label + 1 for new_label, cluster_id in enumerate(data['cluster'].unique())}
data['new_label'] = data['cluster'].map(cluster_labels)

# 更改特定标签的数据为0
data.loc[data['label'].isin(labels_to_remove), 'new_label'] = 0

merged_data = pd.merge(shuju, data, on='orderid', how='inner')




# # 绘制所有点的位置和区域边界
# plt.figure(figsize=(10, 8))
#
# # 绘制数据点
# scatter = plt.scatter(data['longitude'], data['latitude'], c=data['new_label'], cmap='viridis', s=5)
#
# # 给每个类别添加标注
# for new_label, cluster_count in data['new_label'].value_counts().items():
#     if new_label != 0:
#         cluster_data = data[data['new_label'] == new_label]
#         cluster_center = cluster_data[['longitude', 'latitude']].mean().values
#
#         plt.text(cluster_center[0], cluster_center[1], str(cluster_count), color='black', fontsize=10,
#                  ha='center', va='center')

# ...（绘制区域边界的部分保持不变）

# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('KMeans Clustering with Cluster Sizes and Updated Labels')
# plt.colorbar(scatter, label='Cluster')
# plt.show()


# 打印每个类别的点数（按顺序）
cluster_sizes = data['new_label'].value_counts().sort_index()
for new_label, cluster_size in cluster_sizes.items():
    print(f'Cluster {new_label}: {cluster_size} points')

dic = {}
# 打印每个类别的中心坐标v
for new_label, cluster_center in enumerate(kmeans.cluster_centers_):
    dic[new_label] = cluster_center
print(dic)




import pickle
with open('dic.pkl', 'wb') as f:
    pickle.dump(dic ,f)


# # 使用apply函数
# merged_data['category'] = merged_data.apply(
#     lambda row: find_nearest_category(
#         (row['geohashed_end_loc_Longitude'], row['geohashed_end_loc_Latitude']),
#         dic
#     ),
#     axis=1
# )
# print(merged_data[['geohashed_end_loc_Latitude', 'geohashed_end_loc_Longitude', 'category']])
#

# 导出最终的数据表
merged_data.to_csv('final_clustered_data_test.csv', index=False)


