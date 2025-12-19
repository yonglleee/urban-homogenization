'''
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from scipy.spatial.distance import cosine

def compute_degrees_and_weighted_degrees_from_csv(file):
    """
    计算CSV文件中每行样本的degree、带权度（weighted degree）和大于0的相似度。
    - 度（degree）定义为样本与参考矩阵的余弦相似度大于 0 的数量。
    - 带权度（weighted degree）定义为每行样本与其他样本的相似度之和。
    - 大于0的相似度（positive degree）只计算相似度大于0的样本的总和。
    
    :param csv_file: 输入CSV文件路径
    :return: 返回更新后的 DataFrame，新增 'degree', 'weighted_degree', 'positive_degree' 和相应均值列
    """
    # 1. 读取CSV文件
    print("读取CSV文件...")
    # df = pd.read_pickle(file).head()  # 只读取前5行进行测试
    df = pd.read_pickle(file)  # 全部数据
    # 去掉之前的结果 计算当前新的
    target_columns = ['degree', 'weighted_degree', 'positive_degree', 'weighted_degree_avg', 'positive_degree_avg']
    if any(col in df.columns for col in target_columns):
        # 如果存在，删除这些列
        df = df.drop(columns=[col for col in target_columns if col in df.columns])

    print(f"CSV文件读取完成，包含 {df.shape[0]} 行，{df.shape[1]} 列。")
    
    # 2. 提取特征列（假设列名是 'feature_0', 'feature_1', ..., 'feature_767'）
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    print(f'找到 {len(feature_cols)} 个特征列：{feature_cols[:10]}...')  # 打印前10个特征列名
    if not feature_cols:
        raise ValueError("DataFrame中未找到特征列。")
    
    # 3. 提取特征并标准化
    print("提取特征并进行标准化...")
    features = df[feature_cols].values

    
    # 4. 转换为 PyTorch Tensor，并转移到 GPU
    print("将特征转换为 PyTorch Tensor 并转移到GPU...")
    feature_matrix = torch.tensor(features, dtype=torch.float32).cuda()
    feature_norm = torch.norm(feature_matrix, p=2, dim=1, keepdim=True)
    print("标准化完成")
    # 对特征矩阵进行归一化
    normalized_features = feature_matrix / feature_norm
    print("转移到GPU完成")
    
    # 5. 初始化度列表、带权度列表、大于0相似度的列表
    degrees = []
    weighted_degrees = []
    positive_degrees = []
    weighted_degrees_avg = []
    positive_degrees_avg = []

    num_samples = feature_matrix.size(0)
    print(f"开始计算度数指标，共 {num_samples} 个样本。")
    
    # 6. 逐行计算度、带权度和大于0相似度
    for i in tqdm(range(num_samples), desc="Calculating Degrees"):
        row_tensor = feature_matrix[i]

        # 计算余弦相似度
        similarities = F.cosine_similarity(normalized_features, row_tensor.unsqueeze(0), dim=1)
        if i == 1:
            print(similarities)
        # 计算degree：相似度大于0的数量
        degree = (similarities > 0.9).sum().item()
        
        # 带权度：所有相似度的总和
        weighted_degree = similarities.sum().item()
        
        # 大于0的相似度：相似度大于0的总和
        positive_degree = similarities[similarities > 0.9].sum().item()
        
        # 计算均值
        weighted_degree_avg = weighted_degree / num_samples
        positive_degree_avg = positive_degree / degree if degree > 0 else 0
        
        # 将计算结果添加到列表中
        degrees.append(degree)
        weighted_degrees.append(weighted_degree)
        positive_degrees.append(positive_degree)
        weighted_degrees_avg.append(weighted_degree_avg)
        positive_degrees_avg.append(positive_degree_avg)
    
    print("度数指标计算完成。")
    
    # 7. 添加新的列到 DataFrame
    print("将计算结果添加到DataFrame中...")
    df['degree'] = degrees
    df['weighted_degree'] = weighted_degrees
    df['positive_degree'] = positive_degrees
    df['weighted_degree_avg'] = weighted_degrees_avg
    df['positive_degree_avg'] = positive_degrees_avg
    print("新列添加完成。")
    
    return df

# 示例调用
if __name__ == "__main__":
    root = '/home/liyong/code/CityHomogeneity/data/baidu/V21_mask/'
    # p_file = root + 'meta_test_loss_feature_mask_degree.p'
    p_file = '/home/liyong/code/CityHomogeneity/data/baidu/V21/test_loss_and_feature.pkl'
    print("开始计算度数和带权度...")
    df_with_degrees = compute_degrees_and_weighted_degrees_from_csv(p_file)
    
    # 如果需要，将结果保存为新的 pickle 文件
    out_p_file = root + 'meta_test_loss_feature_mask_degree_09.p'
    print(f"将结果保存为Pickle文件：{out_p_file}")
    df_with_degrees.to_pickle(out_p_file)
    print("保存完成。")
'''

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

# 路径定义
root = '/home/liyong/code/CityHomogeneity/'
shp_root = root + 'data/UrbanBoundaries/preprocess_city49/clipped_gub_county/'
shp_boundary_list = [
    'clipped_gub_county_1990.shp',
    'expanded_areas_1995_1990.shp',
    'expanded_areas_2000_1995.shp',
    'expanded_areas_2005_2000.shp',
    'expanded_areas_2010_2005.shp',
    'expanded_areas_2015_2010.shp',
    'expanded_areas_2018_2015.shp'
]
pkl_file = root + 'output/baidu/V3/train_loss_feature_allcity_300ep.pkl'
output_file = root + 'output/baidu/V3/allcity_train_loss_metric_statistics.xlsx'

# 加载 .pkl 文件
print("Loading data file...")
df_data = pd.read_pickle(pkl_file)
geometry = [Point(xy) for xy in zip(df_data['longitude'], df_data['latitude'])]
gdf_data = gpd.GeoDataFrame(df_data, geometry=geometry, crs='EPSG:4326')

# 创建 Excel 写入器
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for shp_file in shp_boundary_list:
        shp_path = shp_root + shp_file
        print(f"Processing {shp_file}...")

        # 加载 Shapefile 文件
        gdf_shp = gpd.read_file(shp_path)

        # 结果存储列表
        result = []

        # 遍历每个区域
        for _, region in gdf_shp.iterrows():
            name_1 = region['NAME_1']
            name_2 = region['NAME_2']
            name_3 = region['NAME_3']

            # 筛选落在当前区域范围内的点
            region_geom = region['geometry']
            points_in_region = gdf_data[gdf_data.geometry.within(region_geom)]

            # 计算统计值
            if not points_in_region.empty:
                loss_values = points_in_region['loss']
                loss_mean = loss_values.mean()
                loss_var = loss_values.var()
                loss_median = loss_values.median()
                loss_Q1 = np.percentile(loss_values, 25)
                loss_Q3 = np.percentile(loss_values, 75)
                loss_min = loss_values.min()
                loss_max = loss_values.max()
            else:
                loss_mean = float('nan')
                loss_var = float('nan')
                loss_median = float('nan')
                loss_Q1 = float('nan')
                loss_Q3 = float('nan')
                loss_min = float('nan')
                loss_max = float('nan')

            # 添加结果
            result.append({
                'data': shp_file,
                'NAME_1': name_1,
                'NAME_2': name_2,
                'NAME_3': name_3,
                'loss_mean': loss_mean,
                'loss_var': loss_var,
                'loss_median': loss_median,
                'loss_Q1': loss_Q1,
                'loss_Q3': loss_Q3,
                'loss_min': loss_min,
                'loss_max': loss_max
            })

        # 保存当前 Shapefile 的结果到 Excel 表中
        result_df = pd.DataFrame(result)
        sheet_name = shp_file.split('.')[0]  # 用文件名作为工作表名
        result_df.to_excel(writer, index=False, sheet_name=sheet_name)

print(f"所有统计结果已保存到 {output_file}")
