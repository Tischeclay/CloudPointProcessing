#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云噪点去除程序使用示例
演示如何使用不同的滤波方法
"""

from point_cloud_denoising import PointCloudDenoiser
import os

def create_sample_point_cloud():
    """创建一个示例点云文件用于测试"""
    import numpy as np
    
    # 生成一个球形点云
    np.random.seed(42)
    n_points = 1000
    
    # 主球体
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = np.random.normal(1, 0.1, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # 添加一些噪点
    noise_points = 50
    noise_x = np.random.uniform(-2, 2, noise_points)
    noise_y = np.random.uniform(-2, 2, noise_points)
    noise_z = np.random.uniform(-2, 2, noise_points)
    
    # 合并点云
    all_x = np.concatenate([x, noise_x])
    all_y = np.concatenate([y, noise_y])
    all_z = np.concatenate([z, noise_z])
    
    # 保存为.xyz文件
    points = np.column_stack([all_x, all_y, all_z])
    np.savetxt('sample_point_cloud.xyz', points, fmt='%.6f %.6f %.6f')
    
    print(f"已创建示例点云文件: sample_point_cloud.xyz")
    print(f"总点数: {len(points)} (包含 {noise_points} 个噪点)")

def demonstrate_different_methods():
    """演示不同滤波方法的效果"""
    
    # 创建示例点云
    create_sample_point_cloud()
    
    # 测试不同方法
    methods = ['statistical', 'radius', 'clustering', 'density', 'combined']
    
    for method in methods:
        print(f"\n=== 测试 {method} 滤波方法 ===")
        
        # 创建噪点去除器
        denoiser = PointCloudDenoiser()
        
        # 加载数据
        if not denoiser.load_xyz_file('sample_point_cloud.xyz'):
            continue
        
        # 应用滤波
        if method == 'statistical':
            denoiser.statistical_outlier_removal(nb_neighbors=20, std_ratio=2.0)
        elif method == 'radius':
            denoiser.radius_outlier_removal(radius=0.5, min_points=5)
        elif method == 'clustering':
            denoiser.clustering_denoising(eps=0.3, min_samples=8)
        elif method == 'density':
            denoiser.density_based_denoising(k=10, density_threshold=0.5)
        elif method == 'combined':
            denoiser.combined_denoising()
        
        # 保存结果
        output_file = f'filtered_{method}.xyz'
        denoiser.save_xyz_file(output_file)
        
        # 显示统计信息
        stats = denoiser.get_statistics()
        if stats:
            print(f"原始点数: {stats['original_points']}")
            print(f"保留点数: {stats['filtered_points']}")
            print(f"移除点数: {stats['removed_points']}")
            print(f"保留率: {stats['retention_rate']:.2f}%")

def batch_process_example():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 如果没有示例文件，先创建一个
    if not os.path.exists('sample_point_cloud.xyz'):
        create_sample_point_cloud()
    
    # 使用组合方法进行批量处理
    denoiser = PointCloudDenoiser()
    
    if denoiser.load_xyz_file('sample_point_cloud.xyz'):
        # 应用组合滤波
        custom_params = {
            'statistical': {'nb_neighbors': 25, 'std_ratio': 2.0},
            'radius': {'radius': 0.5, 'min_points': 5},
            'clustering': {'eps': 0.3, 'min_samples': 10}
        }
        
        denoiser.combined_denoising(custom_params)
        
        # 保存结果
        denoiser.save_xyz_file('batch_processed.xyz')
        
        # 生成可视化对比图
        denoiser.visualize_results(save_plots=True)
        
        print("批量处理完成!")

if __name__ == "__main__":
    print("点云噪点去除程序使用示例")
    print("=" * 50)
    
    # 演示不同方法
    demonstrate_different_methods()
    
    # 批量处理示例
    batch_process_example()
    
    print("\n示例运行完成!")
    print("生成的文件:")
    for file in ['sample_point_cloud.xyz', 'filtered_statistical.xyz', 
                 'filtered_radius.xyz', 'filtered_clustering.xyz', 
                 'filtered_density.xyz', 'filtered_combined.xyz',
                 'batch_processed.xyz', 'point_cloud_comparison.png']:
        if os.path.exists(file):
            print(f"  ✓ {file}")