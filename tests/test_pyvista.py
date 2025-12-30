import sys 
import pyvista as pv
import numpy as np
import xml.etree.ElementTree as ET
import os
from pathlib import Path

class TimeSeriesViewer:
    def __init__(self, pvd_file):
        self.vtk_files, self.time_values = self._extract_from_pvd(pvd_file)
        print(f"找到 {len(self.vtk_files)} 个VTK文件")
        
        # 逐个加载文件以避免内存问题
        self.blocks = []
        for i, vtk_file in enumerate(self.vtk_files):
            try:
                mesh = pv.read(vtk_file)
                self.blocks.append(mesh)
                print(f"  [{i+1}/{len(self.vtk_files)}] 加载: {Path(vtk_file).name}")
            except Exception as e:
                print(f"  [!] 加载失败 {vtk_file}: {e}")
                # 添加空网格作为占位符
                self.blocks.append(pv.UnstructuredGrid())
        
        self.data = pv.MultiBlock(self.blocks)
        
        if not self.time_values:
            self.time_values = list(range(len(self.data)))
        
        self.current_idx = 0
        self.num_steps = len(self.data)
        
        # 获取所有可用的场（检查前5个时间步）
        self.available_fields = self._get_available_fields()
        
        # 当前选择
        self.current_field = None
        self.current_cmap = 'viridis'
        self.show_edges = True
        self.is_playing = False
        self.play_speed = 1.0  # 播放速度倍数
        
        print(f"\n加载成功！共 {self.num_steps} 个时间步")
        print(f"时间范围: {self.time_values[0]} 到 {self.time_values[-1]}")
        print(f"可用场: {self.available_fields}")
        
        # 设置默认场
        if self.available_fields:
            self.current_field = self.available_fields[0]
        
        # 创建绘图器
        self.plotter = pv.Plotter(window_size=(1400, 900))
        
        # 设置UI
        self._setup_ui()
        
        # 初始显示
        self._display_current_step()
    
    def _extract_from_pvd(self, pvd_file):
        """从PVD文件中提取时间值和文件路径"""
        pvd_dir = os.path.dirname(pvd_file)            
        tree = ET.parse(pvd_file)
        root = tree.getroot()
        
        time_values = []
        vtk_files = []
        
        collection = root.find('.//Collection')
        if collection is None:
            print("警告: PVD文件中没有Collection节点")
            return vtk_files, time_values
        
        for dataset in collection:
            if dataset.tag == 'DataSet':
                timestep = dataset.get('timestep')
                file = dataset.get('file')
                if file:
                    # 处理文件路径
                    full_path = os.path.join(pvd_dir, file)
                    # 检查文件是否存在
                    if os.path.exists(full_path):
                        vtk_files.append(full_path)
                        if timestep:
                            time_values.append(float(timestep))
                        else:
                            time_values.append(len(time_values))  # 使用索引作为时间
                    else:
                        print(f"警告: 文件不存在 - {full_path}")
        
        return vtk_files, time_values
    
    def _get_available_fields(self):
        """获取所有可用的场数据"""
        fields_set = set()
        
        # 检查前几个时间步获取所有可能的场
        for i in range(min(5, len(self.blocks))):
            mesh = self.blocks[i]
            if hasattr(mesh, 'array_names'):
                for field in mesh.array_names:
                    # 过滤掉几何相关的字段
                    if field not in ['Normals', 'TCoords', 'TextureCoordinates']:
                        fields_set.add(field)
        
        # 按优先级排序
        priority_fields = [
            'Velocity', 'velocity', 'v', 'vel',
            'Pressure', 'pressure', 'p',
            'Temperature', 'temperature', 'T',
            'Density', 'density', 'rho',
            'Vorticity', 'vorticity',
            'Displacement', 'displacement',
            'mag', 'magnitude', 'Mag', 'Magnitude'
        ]
        
        sorted_fields = []
        # 添加优先级高的字段
        for field in priority_fields:
            if field in fields_set:
                sorted_fields.append(field)
                fields_set.remove(field)
        
        # 添加剩余的字段
        sorted_fields.extend(sorted(fields_set))
        
        return sorted_fields
    
    def _setup_ui(self):
        """设置用户界面"""
        # 设置背景和标题
        self.plotter.set_background('blue')
        
        # 1. 时间控制区域
        self._setup_time_controls()
        
        # 2. 场选择区域
        self._setup_field_controls()
        
        # 3. 显示选项区域
        self._setup_display_controls()
        
        # 4. 添加键盘快捷键
        self._setup_keyboard_shortcuts()
    
    def _setup_time_controls(self):
        """设置时间控制UI"""
        # 主时间滑块
        self.time_slider = self.plotter.add_slider_widget(
            self._on_time_slider_changed,
            [0, self.num_steps-1],
            value=0,
            title="时间步",
            pointa=(0.02, 0.12),
            pointb=(0.35, 0.12),
            style='modern',
            interaction_event='always',
            tube_width=0.02,
            slider_width=0.05
        )
        
        # 播放速度滑块
        self.speed_slider = self.plotter.add_slider_widget(
            self._on_speed_changed,
            [0.1, 5.0],
            value=1.0,
            title="播放速度",
            pointa=(0.02, 0.07),
            pointb=(0.18, 0.07),
            style='modern',
            fmt="%.1f"
        )
        
        # 播放/暂停按钮
        self.play_button = self.plotter.add_checkbox_button_widget(
            self._on_play_toggled,
            value=False,
            position=(0.2, 0.065),
            size=35,
            color_on='green',
            color_off='red',
            background_color='gray'
        )
        
        # 添加播放按钮文本
        self.plotter.add_text(
            "Start",
            position=(0.235, 0.07),
            font_size=10,
            color='white'
        )
        
        # 步进按钮
        button_size = 30
        button_y = 0.065
        
        # 上一帧按钮
        self.plotter.add_checkbox_button_widget(
            self._prev_frame,
            position=(0.28, button_y),
            size=button_size,
            color_on='blue'
        )
        self.plotter.add_text("◀", position=(0.295, button_y+0.005), font_size=12, color='white')
        
        # 下一帧按钮
        self.plotter.add_checkbox_button_widget(
            self._next_frame,
            position=(0.32, button_y),
            size=button_size,
            color_on='blue'
        )
        self.plotter.add_text("▶", position=(0.335, button_y+0.005), font_size=12, color='white')
    
    def _setup_field_controls(self):
        """设置场选择UI"""
        if not self.available_fields:
            return
        
        # 场选择下拉菜单
        self.plotter.add_text(
            "Choose field:",
            position=(0.85, 0.95),
            font_size=10,
            color='white'
        )
        # self.field_selector = self.plotter.add_checkbox_button_widget(
        #     self._on_field_selected,
        #     self.available_fields,
        #     value=self.current_field if self.current_field else self.available_fields[0],
        #     pointa=(0.85, 0.90),
        #     pointb=(0.98, 0.90),
        #     font_size=10
        # )
    
    def _setup_display_controls(self):
        """设置显示选项UI"""
        # 颜色映射选择
        colormaps = ['viridis', 'plasma', 'coolwarm', 'jet', 'rainbow', 
                    'hot', 'gray', 'bone', 'winter', 'spring']
        
        self.plotter.add_text(
            "Color map:",
            position=(0.85, 0.85),
            font_size=10,
            color='white'
        )
        
        # self.cmap_selector = self.plotter.add_dropdown_widget(
        #     self._on_cmap_selected,
        #     colormaps,
        #     value='viridis',
        #     pointa=(0.85, 0.80),
        #     pointb=(0.98, 0.80),
        #     font_size=10
        # )
        
        # 边缘显示切换
        self.edge_toggle = self.plotter.add_checkbox_button_widget(
            self._on_edges_toggled,
            value=self.show_edges,
            position=(0.85, 0.75),
            size=25,
            border_size=2
        )
        
        self.plotter.add_text(
            "Show border",
            position=(0.88, 0.75),
            font_size=10,
            color='white'
        )
        
        # 颜色范围自动调整
        self.auto_clim = True
        self.clim_toggle = self.plotter.add_checkbox_button_widget(
            self._on_clim_toggled,
            value=self.auto_clim,
            position=(0.85, 0.70),
            size=25,
            border_size=2
        )
        
        self.plotter.add_text(
            "Auto color range",
            position=(0.88, 0.70),
            font_size=10,
            color='white'
        )
    
    def _setup_keyboard_shortcuts(self):
        """设置键盘快捷键"""
        # 空格键：播放/暂停
        self.plotter.add_key_event("space", self._toggle_playback)
        
        # 左右箭头：前后帧
        self.plotter.add_key_event("Right", self._next_frame)
        self.plotter.add_key_event("Left", self._prev_frame)
        
        # +/-：调整播放速度
        self.plotter.add_key_event("plus", lambda: self._adjust_speed(0.1))
        self.plotter.add_key_event("minus", lambda: self._adjust_speed(-0.1))
        
        # F键：切换场
        self.plotter.add_key_event("f", self._cycle_field)
        
        # R键：重置视图
        self.plotter.add_key_event("r", self._reset_view)
    
    # ========== 回调函数 ==========
    
    def _on_time_slider_changed(self, value):
        """时间滑块回调"""
        self.current_idx = int(value)
        self._display_current_step()
    
    def _on_field_selected(self, field):
        """场选择回调"""
        self.current_field = field
        print(f"选择场: {field}")
        self._display_current_step()
    
    def _on_cmap_selected(self, cmap):
        """颜色映射回调"""
        self.current_cmap = cmap
        self._display_current_step()
    
    def _on_edges_toggled(self, state):
        """边缘显示回调"""
        self.show_edges = bool(state)
        self._display_current_step()
    
    def _on_clim_toggled(self, state):
        """颜色范围回调"""
        self.auto_clim = bool(state)
        self._display_current_step()
    
    def _on_play_toggled(self, state):
        """播放/暂停回调"""
        self.is_playing = bool(state)
        if self.is_playing:
            self._play_animation()
    
    def _on_speed_changed(self, value):
        """播放速度回调"""
        self.play_speed = value
    
    def _toggle_playback(self):
        """切换播放状态"""
        self.is_playing = not self.is_playing
        self.play_button.GetRepresentation().SetValue(self.is_playing)
        if self.is_playing:
            self._play_animation()
    
    def _next_frame(self):
        """下一帧"""
        if self.current_idx < self.num_steps - 1:
            self.current_idx += 1
            self.time_slider.GetRepresentation().SetValue(self.current_idx)
            self._display_current_step()
    
    def _prev_frame(self):
        """上一帧"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.time_slider.GetRepresentation().SetValue(self.current_idx)
            self._display_current_step()
    
    def _adjust_speed(self, delta):
        """调整播放速度"""
        new_speed = max(0.1, min(5.0, self.play_speed + delta))
        self.play_speed = new_speed
        self.speed_slider.GetRepresentation().SetValue(new_speed)
        print(f"播放速度: {self.play_speed:.1f}")
    
    def _cycle_field(self):
        """循环切换场"""
        if not self.available_fields:
            return
        
        current_index = self.available_fields.index(self.current_field) if self.current_field in self.available_fields else 0
        next_index = (current_index + 1) % len(self.available_fields)
        self.current_field = self.available_fields[next_index]
        self.field_selector.GetRepresentation().SetValue(self.current_field)
        self._display_current_step()
    
    def _reset_view(self):
        """重置视图"""
        self.plotter.reset_camera()
        self.plotter.render()
    
    def _play_animation(self):
        """播放动画"""
        import time
        
        while self.is_playing and self.current_idx < self.num_steps - 1:
            start_time = time.time()
            
            self.current_idx += 1
            self.time_slider.GetRepresentation().SetValue(self.current_idx)
            self._display_current_step()
            
            # 控制播放速度
            elapsed = time.time() - start_time
            sleep_time = max(0.01, 0.1 / self.play_speed - elapsed)
            time.sleep(sleep_time)
            
            # 更新显示
            self.plotter.update()
        
        # 播放结束
        if self.current_idx >= self.num_steps - 1:
            self.is_playing = False
            self.play_button.GetRepresentation().SetValue(False)
    
    def _display_current_step(self):
        """显示当前时间步"""
        # 清除之前的网格
        self.plotter.clear()
        
        if self.current_idx >= len(self.blocks):
            print(f"错误: 索引 {self.current_idx} 超出范围")
            return
        
        mesh = self.blocks[self.current_idx]
        
        if mesh.n_points == 0:
            print(f"警告: 时间步 {self.current_idx} 为空网格")
            return
        
        # 确定要显示的场
        scalars = None
        if self.current_field and self.current_field in mesh.array_names:
            scalars = self.current_field
        elif mesh.array_names:
            # 如果没有选择场，使用第一个可用的
            for field in mesh.array_names:
                if field not in ['Normals', 'TCoords']:
                    scalars = field
                    break
        
        # 确定颜色范围
        clim = None
        if scalars and not self.auto_clim:
            data = mesh[scalars]
            if data is not None and len(data) > 0:
                clim = [np.min(data), np.max(data)]
        
        # 添加网格
        actor = self.plotter.add_mesh(
            mesh,
            scalars=scalars,
            show_edges=self.show_edges and mesh.n_cells < 10000,
            cmap=self.current_cmap,
            clim=clim,
            scalar_bar_args={
                'title': scalars if scalars else 'Value',
                'vertical': True,
                'title_font_size': 12,
                'label_font_size': 10,
                'shadow': True,
                'n_labels': 5
            },
            lighting=True,
            smooth_shading=True,
            ambient=0.3,
            specular=0.5,
            specular_power=30,
            nan_color='gray'
        )
        
        # 添加坐标轴
        self.plotter.add_axes(
            line_width=5,
            cone_radius=0.4,
            shaft_length=0.8,
            tip_length=0.2,
            ambient=0.5,
            label_size=(0.3, 0.3)
        )
        
        # 添加时间信息
        time_text = (
            f"Step: {self.current_idx}/{self.num_steps-1}\n"
            f"Time: {self.time_values[self.current_idx]:.6f}"
        )
        
        self.plotter.add_text(
            time_text,
            position='upper_left',
            font_size=12,
            color='white',
            font='arial',
            shadow=True
        )
        
        # 添加网格信息
        info_text = (
            f"N points: {mesh.n_points:,}\n"
            f"N cells: {mesh.n_cells:,}\n"
            f"Field: {scalars if scalars else '无'}"
        )
        
        self.plotter.add_text(
            info_text,
            position='lower_left',
            font_size=10,
            color='white',
            font='arial'
        )
        
        # 添加标题
        title = f"Time serie inspector - {Path(self.vtk_files[self.current_idx]).name}"
        self.plotter.add_title(title, font_size=14, color='cyan')
        
        # 重置相机
        self.plotter.reset_camera()
        
        # 立即渲染
        self.plotter.render()
    
    def show(self):
        """显示可视化窗口"""
        print("\n" + "="*60)
        print("控制说明:")
        print("  时间滑块: 选择时间步")
        print("  场选择: 下拉菜单选择显示场")
        print("  颜色映射: 下拉菜单选择颜色方案")
        print("  播放/暂停: 按钮或空格键")
        print("  速度控制: 滑块调整播放速度")
        print("  上一帧/下一帧: 按钮或左右箭头键")
        print("  显示边缘: 复选框开关")
        print("  自动颜色范围: 复选框开关")
        print("\n键盘快捷键:")
        print("  空格: 播放/暂停")
        print("  →/←: 下一帧/上一帧")
        print("  +/-: 增加/减小播放速度")
        print("  F: 循环切换场")
        print("  R: 重置视图")
        print("="*60 + "\n")
        
        self.plotter.show()

# 使用示例
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python script.py <pvd文件>")
        print("示例: python script.py simulation.pvd")
        sys.exit(1)
    
    pvd = sys.argv[1]
    
    try:
        viewer = TimeSeriesViewer(pvd)
        viewer.show()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()