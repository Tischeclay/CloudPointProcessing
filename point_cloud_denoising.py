#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三维声呐点云噪点去除核心模块
提供多种点云去噪算法的实现
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class PointCloudDenoiser:
    """点云噪点去除器"""
    
    def __init__(self):
        self.original_cloud = None
        self.filtered_cloud = None
        self.statistics = {}
    
    def load_xyz_file(self, file_path):
        """加载XYZ格式的点云文件"""
        try:
            # 读取XYZ文件
            points = np.loadtxt(file_path)
            
            # 创建Open3D点云对象
            self.original_cloud = o3d.geometry.PointCloud()
            self.original_cloud.points = o3d.utility.Vector3dVector(points)
            
            # 初始化过滤后的点云
            self.filtered_cloud = None
            self.statistics = {}
            
            print(f"成功加载点云文件: {file_path}")
            print(f"点云包含 {len(points)} 个点")
            
            return True
            
        except Exception as e:
            print(f"加载文件失败: {str(e)}")
            return False
    
    def save_xyz_file(self, file_path):
        """保存点云为XYZ格式"""
        if self.filtered_cloud is None:
            print("没有可保存的过滤后点云")
            return False
        
        try:
            points = np.asarray(self.filtered_cloud.points)
            np.savetxt(file_path, points, fmt='%.6f %.6f %.6f')
            print(f"点云已保存到: {file_path}")
            return True
        except Exception as e:
            print(f"保存文件失败: {str(e)}")
            return False
    
    def statistical_outlier_removal(self, nb_neighbors=20, std_ratio=2.0):
        """统计滤波去除离群点"""
        if self.original_cloud is None:
            print("请先加载点云文件")
            return False
        
        try:
            print(f"执行统计滤波 - 邻近点数量: {nb_neighbors}, 标准差倍数: {std_ratio}")
            
            # 执行统计滤波
            self.filtered_cloud, _ = self.original_cloud.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            # 计算统计信息
            self._calculate_statistics()
            
            print(f"统计滤波完成，保留了 {len(self.filtered_cloud.points)} 个点")
            return True
            
        except Exception as e:
            print(f"统计滤波失败: {str(e)}")
            return False
    
    def radius_outlier_removal(self, radius=1.0, min_points=5):
        """半径滤波去除离群点"""
        if self.original_cloud is None:
            print("请先加载点云文件")
            return False
        
        try:
            print(f"执行半径滤波 - 搜索半径: {radius}, 最少点数: {min_points}")
            
            # 执行半径滤波
            self.filtered_cloud, _ = self.original_cloud.remove_radius_outlier(
                nb_points=min_points,
                radius=radius
            )
            
            # 计算统计信息
            self._calculate_statistics()
            
            print(f"半径滤波完成，保留了 {len(self.filtered_cloud.points)} 个点")
            return True
            
        except Exception as e:
            print(f"半径滤波失败: {str(e)}")
            return False
    
    def clustering_denoising(self, eps=0.5, min_samples=10):
        """基于聚类的噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云文件")
            return False
        
        try:
            print(f"执行聚类滤波 - 聚类距离: {eps}, 最少样本: {min_samples}")
            
            # 获取点云坐标
            points = np.asarray(self.original_cloud.points)
            
            # 执行DBSCAN聚类
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            
            # 保留非噪点聚类
            valid_mask = labels != -1
            
            # 创建过滤后的点云
            self.filtered_cloud = o3d.geometry.PointCloud()
            self.filtered_cloud.points = o3d.utility.Vector3dVector(points[valid_mask])
            
            # 计算统计信息
            self._calculate_statistics()
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"聚类滤波完成，发现 {n_clusters} 个聚类，保留了 {len(self.filtered_cloud.points)} 个点")
            return True
            
        except Exception as e:
            print(f"聚类滤波失败: {str(e)}")
            return False
    
    def density_based_denoising(self, k=10, density_threshold=0.1):
        """基于密度的噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云文件")
            return False
        
        try:
            print(f"执行密度滤波 - 邻居数量: {k}, 密度阈值: {density_threshold}")
            
            # 获取点云坐标
            points = np.asarray(self.original_cloud.points)
            
            # 计算每个点的k近邻距离
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
            distances, _ = nbrs.kneighbors(points)
            
            # 使用第k个邻居的距离作为密度指标
            k_distances = distances[:, k]
            
            # 计算密度统计信息
            density_mean = np.mean(k_distances)
            density_std = np.std(k_distances)
            
            # 密度阈值
            threshold = density_mean - density_threshold * density_std
            
            # 保留密度高于阈值的点
            valid_mask = k_distances <= threshold
            
            # 创建过滤后的点云
            self.filtered_cloud = o3d.geometry.PointCloud()
            self.filtered_cloud.points = o3d.utility.Vector3dVector(points[valid_mask])
            
            # 计算统计信息
            self._calculate_statistics()
            
            print(f"密度滤波完成，保留了 {len(self.filtered_cloud.points)} 个点")
            return True
            
        except Exception as e:
            print(f"密度滤波失败: {str(e)}")
            return False
    
    def combined_denoising(self):
        """组合多种滤波方法"""
        if self.original_cloud is None:
            print("请先加载点云文件")
            return False
        
        try:
            print("执行组合滤波...")
            
            # 步骤1: 统计滤波
            temp_cloud, _ = self.original_cloud.remove_statistical_outlier(
                nb_neighbors=15,
                std_ratio=2.0
            )
            print(f"步骤1 - 统计滤波: {len(temp_cloud.points)} 个点")
            
            # 步骤2: 半径滤波
            temp_cloud, _ = temp_cloud.remove_radius_outlier(
                nb_points=5,
                radius=1.0
            )
            print(f"步骤2 - 半径滤波: {len(temp_cloud.points)} 个点")
            
            # 步骤3: 聚类滤波
            points = np.asarray(temp_cloud.points)
            if len(points) > 100:  # 确保有足够的点进行聚类
                clustering = DBSCAN(eps=0.3, min_samples=5).fit(points)
                labels = clustering.labels_
                valid_mask = labels != -1
                
                self.filtered_cloud = o3d.geometry.PointCloud()
                self.filtered_cloud.points = o3d.utility.Vector3dVector(points[valid_mask])
                print(f"步骤3 - 聚类滤波: {len(self.filtered_cloud.points)} 个点")
            else:
                self.filtered_cloud = temp_cloud
            
            # 计算统计信息
            self._calculate_statistics()
            
            print(f"组合滤波完成，最终保留了 {len(self.filtered_cloud.points)} 个点")
            return True
            
        except Exception as e:
            print(f"组合滤波失败: {str(e)}")
            return False
    
    def _calculate_statistics(self):
        """计算处理统计信息"""
        if self.original_cloud is None or self.filtered_cloud is None:
            return
        
        original_count = len(self.original_cloud.points)
        filtered_count = len(self.filtered_cloud.points)
        removed_count = original_count - filtered_count
        retention_rate = (filtered_count / original_count) * 100
        
        self.statistics = {
            'original_points': original_count,
            'filtered_points': filtered_count,
            'removed_points': removed_count,
            'retention_rate': retention_rate
        }
    
    def get_statistics(self):
        """获取处理统计信息"""
        return self.statistics
    
    def visualize_point_clouds(self, show_original=True, show_filtered=True):
        """可视化点云（需要在GUI环境中使用）"""
        try:
            if show_original and self.original_cloud is not None:
                # 创建可视化窗口
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                
                # 添加原始点云
                vis.add_geometry(self.original_cloud)
                
                if show_filtered and self.filtered_cloud is not None:
                    # 为不同点云设置不同的颜色
                    self.original_cloud.paint_uniform_color([1, 0, 0])  # 红色
                    self.filtered_cloud.paint_uniform_color([0, 1, 0])   # 绿色
                    vis.update_geometry(self.filtered_cloud)
                
                # 运行可视化
                vis.run()
                vis.destroy_window()
                
        except Exception as e:
            print(f"可视化失败: {str(e)}")
    
    def export_processing_report(self, file_path):
        """导出处理报告"""
        if not self.statistics:
            print("没有可导出的处理统计信息")
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=== 三维声呐点云噪点去除处理报告 ===\n\n")
                f.write(f"原始点数: {self.statistics['original_points']}\n")
                f.write(f"保留点数: {self.statistics['filtered_points']}\n")
                f.write(f"移除点数: {self.statistics['removed_points']}\n")
                f.write(f"保留率: {self.statistics['retention_rate']:.2f}%\n")
                f.write(f"\n处理完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"处理报告已导出到: {file_path}")
            return True
            
        except Exception as e:
            print(f"导出报告失败: {str(e)}")
            return False

if __name__ == "__main__":
    # 示例使用
    import time
    
    # 创建点云去噪器
    denoiser = PointCloudDenoiser()
    
    print("=== 三维声呐点云噪点去除程序 ===")
    print("支持的文件格式: .xyz")
    print("处理方法: 统计滤波、半径滤波、聚类滤波、密度滤波、组合滤波")
    print("使用方法:")
    print("  denoiser = PointCloudDenoiser()")
    print("  denoiser.load_xyz_file('input.xyz')")
    print("  denoiser.statistical_outlier_removal()")
    print("  denoiser.save_xyz_file('output.xyz')")
    print()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三维声呐点云噪点去除程序
支持直接输入.xyz文件并进行多种噪点去除算法的处理
"""

import numpy as np
import open3d as o3d
import argparse
import os
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudDenoiser:
    def __init__(self):
        self.original_cloud = None
        self.filtered_cloud = None
        
    def load_xyz_file(self, file_path):
        """加载.xyz格式的点云文件"""
        try:
            # 尝试使用open3d直接读取
            cloud = o3d.io.read_point_cloud(file_path, format='xyz')
            if len(cloud.points) > 0:
                self.original_cloud = cloud
                print(f"成功加载点云文件: {file_path}")
                print(f"原始点云点数: {len(cloud.points)}")
                return True
        except:
            pass
            
        try:
            # 如果open3d读取失败，尝试手动解析
            data = np.loadtxt(file_path)
            if data.shape[1] >= 3:
                points = data[:, :3]  # 取前三列作为x,y,z坐标
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                self.original_cloud = cloud
                print(f"成功加载点云文件: {file_path}")
                print(f"原始点云点数: {len(points)}")
                return True
        except Exception as e:
            print(f"加载文件失败: {e}")
            return False
            
        return False
    
    def statistical_outlier_removal(self, nb_neighbors=20, std_ratio=2.0):
        """基于统计的噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云数据")
            return False
            
        # 创建统计滤波器
        filtered_cloud, ind = self.original_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        
        self.filtered_cloud = filtered_cloud
        removed_points = len(self.original_cloud.points) - len(filtered_cloud.points)
        print(f"统计滤波完成，移除了 {removed_points} 个噪点")
        print(f"保留点数: {len(filtered_cloud.points)}")
        return True
    
    def radius_outlier_removal(self, radius=1.0, min_points=5):
        """基于半径的噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云数据")
            return False
            
        # 创建半径滤波器
        filtered_cloud, ind = self.original_cloud.remove_radius_outlier(
            nb_points=min_points, radius=radius
        )
        
        self.filtered_cloud = filtered_cloud
        removed_points = len(self.original_cloud.points) - len(filtered_cloud.points)
        print(f"半径滤波完成，移除了 {removed_points} 个噪点")
        print(f"保留点数: {len(filtered_cloud.points)}")
        return True
    
    def clustering_denoising(self, eps=0.5, min_samples=10):
        """基于聚类的噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云数据")
            return False
            
        # 转换为numpy数组
        points = np.asarray(self.original_cloud.points)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # 保留主聚类的点（标记为-1的是噪点）
        main_cluster_mask = labels != -1
        filtered_points = points[main_cluster_mask]
        
        # 创建新的点云
        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        
        self.filtered_cloud = filtered_cloud
        removed_points = len(points) - len(filtered_points)
        clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = list(labels).count(-1)
        
        print(f"聚类滤波完成，发现 {clusters} 个聚类")
        print(f"噪点数量: {noise_points}")
        print(f"移除了 {removed_points} 个噪点")
        print(f"保留点数: {len(filtered_points)}")
        return True
    
    def density_based_denoising(self, k=10, density_threshold=0.1):
        """基于密度的噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云数据")
            return False
            
        points = np.asarray(self.original_cloud.points)
        n_points = len(points)
        
        # 计算每个点到其k个最近邻点的平均距离
        densities = []
        for i in range(n_points):
            # 计算到其他点的距离
            distances = np.sqrt(np.sum((points[i] - points) ** 2, axis=1))
            # 排除自己，获取k个最近邻
            nearest_distances = np.sort(distances[distances > 0])[:k]
            if len(nearest_distances) > 0:
                avg_distance = np.mean(nearest_distances)
                densities.append(1.0 / avg_distance if avg_distance > 0 else 0)
            else:
                densities.append(0)
        
        densities = np.array(densities)
        
        # 根据密度阈值筛选点
        density_mask = densities >= density_threshold
        filtered_points = points[density_mask]
        
        # 创建新的点云
        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        
        self.filtered_cloud = filtered_cloud
        removed_points = n_points - len(filtered_points)
        
        print(f"密度滤波完成，移除了 {removed_points} 个低密度点")
        print(f"保留点数: {len(filtered_points)}")
        return True
    
    def combined_denoising(self, methods_params=None):
        """组合多种方法进行噪点去除"""
        if self.original_cloud is None:
            print("请先加载点云数据")
            return False
            
        if methods_params is None:
            methods_params = {
                'statistical': {'nb_neighbors': 20, 'std_ratio': 2.0},
                'radius': {'radius': 1.0, 'min_points': 5},
                'clustering': {'eps': 0.5, 'min_samples': 10}
            }
        
        print("开始组合滤波处理...")
        
        # 第一步：统计滤波
        self.statistical_outlier_removal(**methods_params['statistical'])
        if self.filtered_cloud is None:
            return False
            
        # 第二步：半径滤波
        temp_cloud = self.filtered_cloud
        filtered_cloud, ind = temp_cloud.remove_radius_outlier(
            nb_points=methods_params['radius']['min_points'], 
            radius=methods_params['radius']['radius']
        )
        self.filtered_cloud = filtered_cloud
        
        # 第三步：聚类滤波
        points = np.asarray(self.filtered_cloud.points)
        clustering = DBSCAN(eps=methods_params['clustering']['eps'], 
                          min_samples=methods_params['clustering']['min_samples']).fit(points)
        labels = clustering.labels_
        main_cluster_mask = labels != -1
        filtered_points = points[main_cluster_mask]
        
        final_cloud = o3d.geometry.PointCloud()
        final_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        self.filtered_cloud = final_cloud
        
        total_removed = len(self.original_cloud.points) - len(self.filtered_cloud.points)
        print(f"组合滤波完成，总共移除了 {total_removed} 个噪点")
        print(f"最终保留点数: {len(self.filtered_cloud.points)}")
        return True
    
    def save_xyz_file(self, output_path):
        """保存处理后的点云为.xyz文件"""
        if self.filtered_cloud is None:
            print("没有可保存的滤波结果")
            return False
            
        try:
            # 转换为numpy数组
            points = np.asarray(self.filtered_cloud.points)
            # 保存为.xyz格式（只有坐标）
            np.savetxt(output_path, points, fmt='%.6f %.6f %.6f')
            print(f"结果已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False
    
    def visualize_results(self, save_plots=False):
        """可视化原始点云和滤波后的点云"""
        if self.original_cloud is None or self.filtered_cloud is None:
            print("没有足够的点云数据进行可视化")
            return
            
        # 创建可视化窗口
        o3d.visualization.draw_geometries([self.original_cloud], 
                                        window_name="原始点云")
        o3d.visualization.draw_geometries([self.filtered_cloud], 
                                        window_name="滤波后点云")
        
        if save_plots:
            # 使用matplotlib创建对比图
            fig = plt.figure(figsize=(15, 5))
            
            # 原始点云
            ax1 = fig.add_subplot(131, projection='3d')
            orig_points = np.asarray(self.original_cloud.points)
            ax1.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2], 
                       c='red', s=1, alpha=0.6)
            ax1.set_title('原始点云')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 滤波后点云
            ax2 = fig.add_subplot(132, projection='3d')
            filtered_points = np.asarray(self.filtered_cloud.points)
            ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], 
                       c='blue', s=1, alpha=0.6)
            ax2.set_title('滤波后点云')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 差值（被移除的点）
            ax3 = fig.add_subplot(133, projection='3d')
            removed_points = orig_points[~np.isin(orig_points, filtered_points).all(axis=1)]
            ax3.scatter(removed_points[:, 0], removed_points[:, 1], removed_points[:, 2], 
                       c='green', s=1, alpha=0.6)
            ax3.set_title('移除的噪点')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            plt.tight_layout()
            plt.savefig('point_cloud_comparison.png', dpi=300, bbox_inches='tight')
            print("对比图已保存为 point_cloud_comparison.png")
    
    def get_statistics(self):
        """获取点云处理统计信息"""
        if self.original_cloud is None:
            return None
            
        stats = {
            'original_points': len(self.original_cloud.points),
            'filtered_points': len(self.filtered_cloud.points) if self.filtered_cloud else 0,
            'removed_points': len(self.original_cloud.points) - len(self.filtered_cloud.points) if self.filtered_cloud else 0,
            'retention_rate': 0.0
        }
        
        if stats['original_points'] > 0 and self.filtered_cloud:
            stats['retention_rate'] = stats['filtered_points'] / stats['original_points'] * 100
            
        return stats

def main():
    parser = argparse.ArgumentParser(description='三维声呐点云噪点去除程序')
    parser.add_argument('input_file', help='输入的.xyz点云文件路径')
    parser.add_argument('output_file', help='输出的.xyz点云文件路径')
    parser.add_argument('--method', choices=['statistical', 'radius', 'clustering', 'density', 'combined'], 
                       default='combined', help='噪点去除方法 (默认: combined)')
    parser.add_argument('--visualize', action='store_true', help='可视化处理结果')
    parser.add_argument('--save-plots', action='store_true', help='保存对比图')
    parser.add_argument('--params', type=str, help='自定义参数(JSON格式)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在")
        return
    
    # 创建噪点去除器
    denoiser = PointCloudDenoiser()
    
    # 加载点云数据
    if not denoiser.load_xyz_file(args.input_file):
        return
    
    # 处理自定义参数
    methods_params = None
    if args.params:
        import json
        try:
            methods_params = json.loads(args.params)
        except:
            print("警告: 自定义参数格式错误，使用默认参数")
    
    # 根据指定方法进行噪点去除
    success = False
    if args.method == 'statistical':
        success = denoiser.statistical_outlier_removal()
    elif args.method == 'radius':
        success = denoiser.radius_outlier_removal()
    elif args.method == 'clustering':
        success = denoiser.clustering_denoising()
    elif args.method == 'density':
        success = denoiser.density_based_denoising()
    elif args.method == 'combined':
        success = denoiser.combined_denoising(methods_params)
    
    if not success:
        print("噪点去除处理失败")
        return
    
    # 保存结果
    denoiser.save_xyz_file(args.output_file)
    
    # 显示统计信息
    stats = denoiser.get_statistics()
    if stats:
        print("\n=== 处理统计信息 ===")
        print(f"原始点数: {stats['original_points']}")
        print(f"保留点数: {stats['filtered_points']}")
        print(f"移除点数: {stats['removed_points']}")
        print(f"保留率: {stats['retention_rate']:.2f}%")
    
    # 可视化结果
    if args.visualize:
        denoiser.visualize_results(args.save_plots)
    
    print(f"\n处理完成！结果已保存到: {args.output_file}")

if __name__ == "__main__":
    main()