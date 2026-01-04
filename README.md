# 三维声呐点云噪点去除程序

这是一个专为三维声呐扫描数据设计的点云噪点去除工具，提供友好的图形用户界面，支持多种去噪算法。

## 主要特性

### 🖥️ 图形用户界面
- 直观的文件选择界面，支持XYZ格式点云文件
- 实时参数调节滑块，可视化调整去噪参数
- 多方法选择：统计滤波、半径滤波、聚类滤波、密度滤波、组合滤波
- 实时处理进度显示

### 🔧 多种去噪算法
1. **统计滤波 (Statistical Filter)**
   - 基于最近邻统计分析
   - 适合去除随机分布的噪点
   - 参数：邻近点数量、标准差倍数

2. **半径滤波 (Radius Filter)**
   - 基于搜索半径内的点数判断
   - 适合去除孤立噪点
   - 参数：搜索半径、最少点数

3. **聚类滤波 (Clustering Filter)**
   - 基于DBSCAN聚类算法
   - 保留主要聚类，移除小聚类噪点
   - 参数：聚类距离、最少样本数

4. **密度滤波 (Density Filter)**
   - 基于点云局部密度
   - 适合去除低密度噪点
   - 参数：邻居数量、密度阈值

5. **组合滤波 (Combined Filter)**
   - 综合多种算法优势
   - 依次执行统计→半径→聚类滤波
   - 适合复杂噪点场景

### 📊 可视化功能
- 原始点云显示（红色）
- 滤波后点云显示（蓝色）
- 移除噪点显示（绿色）
- 三视图对比显示
- 支持保存对比图

## 安装要求

### 系统要求
- Python 3.7+
- macOS (推荐)
- 内存：至少2GB可用内存
- 存储：至少100MB可用空间

### Python依赖
```bash
pip install -r requirements.txt
```

主要依赖包：
- Open3D >= 0.18.0 - 点云处理核心库
- NumPy >= 1.21.0 - 数值计算
- Scikit-learn >= 1.0.0 - 机器学习算法
- Matplotlib >= 3.5.0 - 数据可视化
- Tkinter - 图形界面（通常随Python安装）

## 使用方法

### 1. 启动程序
```bash
python gui_point_cloud_denoiser.py
```

### 2. 文件操作
- 点击"浏览"按钮选择输入的XYZ文件
- 程序会自动设置输出文件名（可在结果文件名后添加"_denoised"）
- 也可手动设置输出文件路径

### 3. 参数调节
- 选择去噪方法
- 使用滑块调节参数：
  - 邻近点数量：5-50
  - 标准差倍数：0.5-5.0
  - 搜索半径：0.1-5.0
  - 最少点数：1-20
  - 聚类距离：0.1-2.0
  - 最少样本：3-30

### 4. 处理流程
1. 选择输入文件
2. 选择输出文件
3. 调节参数
4. 点击"开始去噪"
5. 等待处理完成
6. 查看可视化结果
7. 保存对比图（可选）

### 5. 结果分析
- 程序会显示处理统计信息
- 原始点数、保留点数、移除点数
- 数据保留率百分比

## XYZ文件格式

输入文件应为标准XYZ格式：
```
x1 y1 z1
x2 y2 z2
x3 y3 z3
...
```

每行包含一个点的X、Y、Z坐标，用空格分隔。

## 使用建议

### 方法选择指南
- **随机噪点**：使用统计滤波
- **孤立噪点**：使用半径滤波
- **聚类噪点**：使用聚类滤波
- **低密度区域噪点**：使用密度滤波
- **复杂噪点**：使用组合滤波

### 参数调优
1. 先使用默认参数进行测试
2. 根据结果调整参数：
   - 保留率过低：减小阈值
   - 保留率过高：增大阈值
3. 对比不同方法的效果
4. 保存最佳参数配置

### 性能优化
- 大文件处理可能需要较长时间
- 建议先处理小样本测试参数
- 关闭其他占用内存的程序

## 故障排除

### 常见问题

**Q: 程序无法启动**
A: 检查Python版本和依赖包安装

**Q: 文件加载失败**
A: 确认文件格式为标准XYZ格式

**Q: 处理速度慢**
A: 尝试减小文件大小或调整参数

**Q: 内存不足**
A: 关闭其他程序或处理更小的文件

**Q: 可视化显示异常**
A: 检查matplotlib后端设置

### 错误信息
- `File not found`: 文件路径错误
- `Invalid file format`: 文件格式不正确
- `Memory error`: 内存不足
- `Processing timeout`: 处理超时

## 技术支持

如遇到问题，请提供：
1. 错误信息截图
2. 文件大小和格式信息
3. 系统环境信息
4. 使用的参数配置

## 版本信息

- 版本：1.0.0
- 更新日期：2024年
- 开发者：AI Assistant
- 许可证：MIT License

## 更新日志

### v1.0.0 (2024-)
- 初始版本发布
- 实现五种去噪算法
- 添加图形用户界面
- 支持XYZ格式输入输出
- 添加可视化功能

---

**注意**: 本程序专为三维声呐数据优化，在其他类型点云数据上的效果可能有所不同。建议根据具体数据特点调整参数。# 三维声呐点云噪点去除程序

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

本程序仅供学习和研究使用。# CloudPointProcessing
