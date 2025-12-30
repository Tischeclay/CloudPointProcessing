# 三维声呐点云噪点去除程序

这是一个专门用于三维声呐扫描点云数据噪点去除的Python程序，支持多种滤波算法并能直接处理.xyz格式文件。

## 功能特点

- **多种滤波算法**：统计滤波、半径滤波、聚类滤波、密度滤波、组合滤波
- **文件格式支持**：直接输入输出.xyz格式文件
- **可视化功能**：支持3D点云可视化和对比图生成
- **参数调节**：支持自定义算法参数
- **处理统计**：提供详细的处理结果统计

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python point_cloud_denoising.py input.xyz output.xyz
```

### 使用指定算法

```bash
# 使用统计滤波
python point_cloud_denoising.py input.xyz output.xyz --method statistical

# 使用聚类滤波
python point_cloud_denoising.py input.xyz output.xyz --method clustering

# 使用组合滤波（默认）
python point_cloud_denoising.py input.xyz output.xyz --method combined
```

### 启用可视化

```bash
# 显示3D可视化窗口
python point_cloud_denoising.py input.xyz output.xyz --visualize

# 保存对比图
python point_cloud_denoising.py input.xyz output.xyz --save-plots
```

### 自定义参数

```bash
# 使用JSON格式自定义参数
python point_cloud_denoising.py input.xyz output.xyz \
  --params '{"statistical": {"nb_neighbors": 30, "std_ratio": 1.5}}'
```

## 算法说明

### 1. 统计滤波 (Statistical)
- 基于点云中每个点与其邻近点的距离分布
- 移除距离异常远的点
- 适用于去除明显的孤立噪点

### 2. 半径滤波 (Radius)
- 基于点的邻域密度
- 移除邻域内点数量过少的点
- 适用于去除稀疏分布的噪点

### 3. 聚类滤波 (Clustering)
- 使用DBSCAN聚类算法
- 保留主要聚类，移除噪点聚类
- 适用于分离不同密度的结构

### 4. 密度滤波 (Density)
- 基于局部密度计算
- 移除密度过低的点
- 适用于保留密集区域

### 5. 组合滤波 (Combined)
- 综合使用多种算法
- 顺序执行统计滤波 → 半径滤波 → 聚类滤波
- 通常能获得最佳效果

## 参数调节建议

### 统计滤波参数
- `nb_neighbors`: 邻近点数量 (建议: 10-50)
- `std_ratio`: 标准差倍数 (建议: 1.0-3.0)

### 半径滤波参数
- `radius`: 搜索半径 (建议: 0.5-2.0)
- `min_points`: 最少邻域点数 (建议: 3-20)

### 聚类滤波参数
- `eps`: 聚类距离阈值 (建议: 0.1-1.0)
- `min_samples`: 最少样本数 (建议: 5-20)

## 输出结果

程序会生成：
- 滤波后的.xyz文件
- 处理统计信息
- 可视化窗口（如果启用）
- 对比图片（如果启用）

## 注意事项

1. 程序主要针对声呐点云数据设计，对其他类型的点云可能需要调整参数
2. 处理大数据集时建议先用小参数测试
3. 可视化功能需要图形界面支持
4. 建议根据具体的声呐数据特性调整算法参数

## 示例

```bash
# 基本处理
python point_cloud_denoising.py sonar_data.xyz cleaned_data.xyz

# 启用可视化并保存对比图
python point_cloud_denoising.py sonar_data.xyz cleaned_data.xyz --visualize --save-plots

# 使用自定义参数
python point_cloud_denoising.py sonar_data.xyz cleaned_data.xyz \
  --method combined \
  --params '{"statistical": {"nb_neighbors": 25, "std_ratio": 2.5}, "clustering": {"eps": 0.3, "min_samples": 8}}'
```

## 许可证

本程序仅供学习和研究使用。