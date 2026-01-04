#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图形界面的三维声呐点云噪点去除程序
提供用户友好的GUI界面进行点云去噪处理
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import threading
import time
from point_cloud_denoising import PointCloudDenoiser

class PointCloudGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("三维声呐点云噪点去除程序")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.input_file_path = tk.StringVar()
        self.output_file_path = tk.StringVar()
        self.denoising_method = tk.StringVar(value="combined")
        self.is_processing = False
        
        # 创建噪点去除器
        self.denoiser = PointCloudDenoiser()
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # 输入文件
        ttk.Label(file_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(file_frame, textvariable=self.input_file_path, state="readonly").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5)
        )
        ttk.Button(file_frame, text="浏览", command=self.select_input_file).grid(
            row=0, column=2, sticky=tk.W
        )
        
        # 输出文件
        ttk.Label(file_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Entry(file_frame, textvariable=self.output_file_path, state="readonly").grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0)
        )
        ttk.Button(file_frame, text="浏览", command=self.select_output_file).grid(
            row=1, column=2, sticky=tk.W, pady=(5, 0)
        )
        
        # 处理参数区域
        param_frame = ttk.LabelFrame(main_frame, text="处理参数", padding="10")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        param_frame.columnconfigure(1, weight=1)
        
        # 去噪方法选择
        ttk.Label(param_frame, text="去噪方法:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        method_frame = ttk.Frame(param_frame)
        method_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        methods = [
            ("统计滤波", "statistical"),
            ("半径滤波", "radius"),
            ("聚类滤波", "clustering"),
            ("密度滤波", "density"),
            ("组合滤波", "combined")
        ]
        
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(method_frame, text=text, variable=self.denoising_method, 
                           value=value).grid(row=0, column=i, sticky=tk.W, padx=(0, 15))
        
        # 参数调节区域
        self.param_frame = ttk.Frame(param_frame)
        self.param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.create_param_controls()
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        self.process_btn = ttk.Button(control_frame, text="开始去噪", 
                                     command=self.start_processing, style="Accent.TButton")
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="可视化结果", 
                  command=self.show_visualization).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="保存对比图", 
                  command=self.save_comparison).pack(side=tk.LEFT, padx=(0, 10))
        
        # 进度条和状态
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=(20, 10))
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="处理结果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 结果文本框
        self.result_text = tk.Text(result_frame, height=8, state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # 可视化区域
        viz_frame = ttk.LabelFrame(main_frame, text="可视化显示", padding="10")
        viz_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置主框架的行权重
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
    def create_param_controls(self):
        """创建参数调节控件"""
        # 清除现有控件
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        method = self.denoising_method.get()
        
        if method == "statistical":
            self.create_statistical_params()
        elif method == "radius":
            self.create_radius_params()
        elif method == "clustering":
            self.create_clustering_params()
        elif method == "density":
            self.create_density_params()
        elif method == "combined":
            self.create_combined_params()
    
    def create_statistical_params(self):
        """创建统计滤波参数控件"""
        self.nb_neighbors_var = tk.IntVar(value=20)
        self.std_ratio_var = tk.DoubleVar(value=2.0)
        
        ttk.Label(self.param_frame, text="邻近点数量:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Scale(self.param_frame, from_=5, to=50, variable=self.nb_neighbors_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(self.param_frame, textvariable=self.nb_neighbors_var).grid(row=0, column=2)
        
        ttk.Label(self.param_frame, text="标准差倍数:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Scale(self.param_frame, from_=0.5, to=5.0, variable=self.std_ratio_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        ttk.Label(self.param_frame, textvariable=self.std_ratio_var).grid(row=1, column=2, pady=(5, 0))
    
    def create_radius_params(self):
        """创建半径滤波参数控件"""
        self.radius_var = tk.DoubleVar(value=1.0)
        self.min_points_var = tk.IntVar(value=5)
        
        ttk.Label(self.param_frame, text="搜索半径:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Scale(self.param_frame, from_=0.1, to=5.0, variable=self.radius_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(self.param_frame, textvariable=self.radius_var).grid(row=0, column=2)
        
        ttk.Label(self.param_frame, text="最少点数:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Scale(self.param_frame, from_=1, to=20, variable=self.min_points_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        ttk.Label(self.param_frame, textvariable=self.min_points_var).grid(row=1, column=2, pady=(5, 0))
    
    def create_clustering_params(self):
        """创建聚类滤波参数控件"""
        self.eps_var = tk.DoubleVar(value=0.5)
        self.min_samples_var = tk.IntVar(value=10)
        
        ttk.Label(self.param_frame, text="聚类距离:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Scale(self.param_frame, from_=0.1, to=2.0, variable=self.eps_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(self.param_frame, textvariable=self.eps_var).grid(row=0, column=2)
        
        ttk.Label(self.param_frame, text="最少样本:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Scale(self.param_frame, from_=3, to=30, variable=self.min_samples_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        ttk.Label(self.param_frame, textvariable=self.min_samples_var).grid(row=1, column=2, pady=(5, 0))
    
    def create_density_params(self):
        """创建密度滤波参数控件"""
        self.k_var = tk.IntVar(value=10)
        self.density_threshold_var = tk.DoubleVar(value=0.1)
        
        ttk.Label(self.param_frame, text="邻居数量:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Scale(self.param_frame, from_=5, to=30, variable=self.k_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(self.param_frame, textvariable=self.k_var).grid(row=0, column=2)
        
        ttk.Label(self.param_frame, text="密度阈值:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Scale(self.param_frame, from_=0.01, to=1.0, variable=self.density_threshold_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        ttk.Label(self.param_frame, textvariable=self.density_threshold_var).grid(row=1, column=2, pady=(5, 0))
    
    def create_combined_params(self):
        """创建组合滤波参数控件"""
        info_label = ttk.Label(self.param_frame, 
                              text="组合滤波将依次执行：统计滤波 → 半径滤波 → 聚类滤波", 
                              font=('TkDefaultFont', 9, 'italic'))
        info_label.grid(row=0, column=0, columnspan=3, sticky=tk.W)
    
    def select_input_file(self):
        """选择输入文件"""
        filename = filedialog.askopenfilename(
            title="选择点云文件",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_path.set(filename)
            # 自动设置输出文件名
            base_name = os.path.splitext(filename)[0]
            self.output_file_path.set(f"{base_name}_denoised.xyz")
            
            # 尝试加载文件以获取基本信息
            try:
                if self.denoiser.load_xyz_file(filename):
                    self.update_status(f"已加载文件: {len(self.denoiser.original_cloud.points)} 个点")
                else:
                    self.update_status("文件加载失败")
            except Exception as e:
                self.update_status(f"文件加载错误: {str(e)}")
    
    def select_output_file(self):
        """选择输出文件"""
        filename = filedialog.asksaveasfilename(
            title="保存处理结果",
            defaultextension=".xyz",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")]
        )
        if filename:
            self.output_file_path.set(filename)
    
    def start_processing(self):
        """开始处理"""
        if not self.input_file_path.get():
            messagebox.showwarning("警告", "请先选择输入文件")
            return
        
        if not self.output_file_path.get():
            messagebox.showwarning("警告", "请先选择输出文件")
            return
        
        if self.is_processing:
            messagebox.showinfo("提示", "正在处理中，请稍候...")
            return
        
        # 在新线程中处理
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.progress.start()
        self.update_status("正在处理...")
        
        thread = threading.Thread(target=self.process_point_cloud)
        thread.daemon = True
        thread.start()
    
    def process_point_cloud(self):
        """处理点云（在线程中执行）"""
        try:
            # 重新加载文件
            if not self.denoiser.load_xyz_file(self.input_file_path.get()):
                self.root.after(0, lambda: messagebox.showerror("错误", "文件加载失败"))
                return
            
            method = self.denoising_method.get()
            success = False
            
            # 根据选择的参数执行不同的滤波方法
            if method == "statistical":
                success = self.denoiser.statistical_outlier_removal(
                    nb_neighbors=self.nb_neighbors_var.get(),
                    std_ratio=self.std_ratio_var.get()
                )
            elif method == "radius":
                success = self.denoiser.radius_outlier_removal(
                    radius=self.radius_var.get(),
                    min_points=self.min_points_var.get()
                )
            elif method == "clustering":
                success = self.denoiser.clustering_denoising(
                    eps=self.eps_var.get(),
                    min_samples=self.min_samples_var.get()
                )
            elif method == "density":
                success = self.denoiser.density_based_denoising(
                    k=self.k_var.get(),
                    density_threshold=self.density_threshold_var.get()
                )
            elif method == "combined":
                success = self.denoiser.combined_denoising()
            
            if success:
                # 保存结果
                self.denoiser.save_xyz_file(self.output_file_path.get())
                
                # 在主线程中更新UI
                self.root.after(0, self.processing_completed)
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "处理失败"))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理出错: {str(e)}"))
        finally:
            # 恢复UI状态
            self.root.after(0, self.reset_ui_state)
    
    def processing_completed(self):
        """处理完成后的操作"""
        # 显示结果统计
        stats = self.denoiser.get_statistics()
        if stats:
            result_msg = f"""
=== 处理完成 ===
处理方法: {self.get_method_name(self.denoising_method.get())}
原始点数: {stats['original_points']}
保留点数: {stats['filtered_points']}
移除点数: {stats['removed_points']}
保留率: {stats['retention_rate']:.2f}%
输出文件: {self.output_file_path.get()}
            """
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result_msg)
            self.result_text.config(state=tk.DISABLED)
            
            # 自动更新可视化
            self.show_visualization()
    
    def reset_ui_state(self):
        """重置UI状态"""
        self.is_processing = False
        self.process_btn.config(state='normal')
        self.progress.stop()
        self.update_status("处理完成")
    
    def get_method_name(self, method):
        """获取方法中文名称"""
        names = {
            "statistical": "统计滤波",
            "radius": "半径滤波",
            "clustering": "聚类滤波",
            "density": "密度滤波",
            "combined": "组合滤波"
        }
        return names.get(method, method)
    
    def update_status(self, message):
        """更新状态信息"""
        self.status_label.config(text=message)
    
    def show_visualization(self):
        """显示可视化结果"""
        if self.denoiser.original_cloud is None or self.denoiser.filtered_cloud is None:
            messagebox.showwarning("警告", "没有可显示的处理结果")
            return
        
        try:
            # 清除之前的图
            self.fig.clear()
            
            # 创建子图
            ax1 = self.fig.add_subplot(131, projection='3d')
            ax2 = self.fig.add_subplot(132, projection='3d')
            ax3 = self.fig.add_subplot(133, projection='3d')
            
            # 原始点云
            orig_points = np.asarray(self.denoiser.original_cloud.points)
            ax1.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2], 
                       c='red', s=1, alpha=0.6)
            ax1.set_title('原始点云')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 滤波后点云
            filtered_points = np.asarray(self.denoiser.filtered_cloud.points)
            ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], 
                       c='blue', s=1, alpha=0.6)
            ax2.set_title('滤波后点云')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 移除的点
            try:
                # 计算移除的点（简化版本）
                removed_mask = np.ones(len(orig_points), dtype=bool)
                for i, point in enumerate(filtered_points):
                    distances = np.linalg.norm(orig_points - point, axis=1)
                    closest_idx = np.argmin(distances)
                    removed_mask[closest_idx] = False
                
                removed_points = orig_points[removed_mask]
                if len(removed_points) > 0:
                    ax3.scatter(removed_points[:, 0], removed_points[:, 1], removed_points[:, 2], 
                               c='green', s=1, alpha=0.6)
                ax3.set_title('移除的噪点')
            except:
                ax3.text(0.5, 0.5, '无法计算移除的点', transform=ax3.transAxes, 
                        ha='center', va='center')
                ax3.set_title('移除的噪点')
            
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            # 调整布局
            self.fig.tight_layout()
            
            # 刷新画布
            self.canvas.draw()
            
            self.update_status("可视化已更新")
            
        except Exception as e:
            messagebox.showerror("错误", f"可视化失败: {str(e)}")
    
    def save_comparison(self):
        """保存对比图"""
        if self.denoiser.original_cloud is None or self.denoiser.filtered_cloud is None:
            messagebox.showwarning("警告", "没有可保存的处理结果")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存对比图",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"对比图已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

def main():
    """主函数"""
    root = tk.Tk()
    app = PointCloudGUI(root)
    
    # 绑定方法变化事件
    app.denoising_method.trace('w', lambda *args: app.create_param_controls())
    
    root.mainloop()

if __name__ == "__main__":
    main()