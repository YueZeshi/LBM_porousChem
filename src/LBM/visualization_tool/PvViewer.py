import sys 
import pyvista as pv
import numpy as np
import xml.etree.ElementTree as ET
import os
from pathlib import Path        
from collections import OrderedDict
from ..util.path import root_path
import ctypes

class PvViewer:
    """基于 PyVista 的 PVD 浏览器。

    功能：
    - 解析 `.pvd` 列表并懒加载各时间步 VTK
    - 交互控件：播放/跳帧/首末帧、速度、帮助、重置相机
    - 自适应 2D/3D 视图与主题配色，支持颜色条与场选择
    """
    def __init__(self, pvd_file):
        """构造可视化窗口并加载 PVD。

        Parameters
        ----------
        pvd_file : str or PathLike
            PVD 文件路径，内部解析出时间与 VTK 文件列表。
        """
        font_path = Path(root_path(), r"font/simhei.ttf")
        self.name = Path(pvd_file).stem
        self.text_kwargs = {"font": "arial"}
        if font_path.exists():
            self.text_kwargs["font_file"] = str(font_path)
            print(f"使用中文字体: {font_path}")
        else:
            print("提示: 未找到中文字体 simhei.ttf，可改成你机器上的中文字体路径")

        pv.global_theme.font.family = "arial"
        pv.global_theme.font.title_size = 12
        pv.global_theme.font.label_size = 10
        # pv.global_theme.font.use_math_text = False

        self.vtk_files, self.time_values = self._extract_from_pvd(pvd_file)
        print(f"找到 {len(self.vtk_files)} 个VTK文件")
        
        # 懒加载：只存路径，实际读取按需 + LRU 缓存
        self.blocks = [None] * len(self.vtk_files)
        self.mesh_cache = OrderedDict()
        self.cache_size = 5

        # 配色方案
        self.palette = {
            'bg': '#444444',
            'bg_top': '#666666',
            'text': '#e5e7eb',
            'muted': '#cbd5e1',
            'highlight': '#67e8f9',
            'accent': '#38bdf8',
            'accent_dark': '#0ea5e9',
            'positive': '#3fb950',
            'primary': '#2563eb',
            'warning': '#eab308',
            'button_on': '#2563ff',
            'button_off': '#aaaaff',
        }

        if not self.time_values:
            self.time_values = list(range(len(self.vtk_files)))
        
        self.current_idx = 0
        self.num_steps = len(self.vtk_files)
        
        # 获取可用场（从前几个时间步探查）
        self.available_fields = self._get_available_fields()
        
        # 当前选择
        self.current_field = self.available_fields[0] if self.available_fields else None
        self.current_cmap = 'jet'
        self.show_edges = True
        self.is_playing = False
        self.play_speed = 1.0  # 播放速度倍数
        
        print(f"\n加载成功！共 {self.num_steps} 个时间步")
        print(f"时间范围: {self.time_values[0]} 到 {self.time_values[-1]}")
        print(f"可用场: {self.available_fields}")
        
        # 创建绘图器，背景渐变，窗口自适应屏幕 80%
        win_w, win_h = self._compute_window_size()
        self.plotter = pv.Plotter(window_size=(win_w, win_h), lighting='light_kit')
        self.plotter.set_background(self.palette['bg'], top=self.palette['bg_top'])
        self.auto_clim = True

        self._mesh_actor = None
        self.bounds = None
        self._last_size = self.plotter.window_size
        # 颜色条布局（右下角）: 归一化坐标
        self._cbar_conf = {
            'position_x': 0.88,
            'position_y': 0.08,
            'width': 0.10,
            'height': 0.36,
        }

        # UI/文本句柄
        self._field_text_actor = None
        self._time_text_actor = None
        self._help_text_actor = None
        self._title_text_override = None  # 使用 add_title
        self._axes_widget_added = False
        self._help_detailed = False
        self.is_helping = False
        # 设置UI
        self._setup_ui()
        # 初始显示
        self._display_current_step()
        self._reset_view()

    

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
        """获取所有可用的场数据（探查前几个时间步，懒加载）"""
        fields_set = set()
        for i in range(min(5, len(self.vtk_files))):
            mesh = self._load_mesh(i, silent=True)
            if mesh is None or mesh.n_points == 0:
                continue
            if hasattr(mesh, 'array_names'):
                for field in mesh.array_names:
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
        """设置用户界面：时间控制、底部按钮、快捷键与帮助文本。"""
        # 1. 时间控制区域（保留滑块）
        self._setup_time_controls()
        # 2. 下方中部按钮
        self._setup_bottom_buttons()
        # 3. 显示选项（保留）
        # self._setup_display_controls()
        # 4. 键盘快捷键
        self._setup_keyboard_shortcuts()
        # 5. 左上角帮助提示（初始为简洁版）
        self._update_help_text()

    def _setup_time_controls(self):
        """设置时间控制UI（使用像素坐标，窗口 80% 自适应）"""
        # 主时间滑块
        self.time_slider = self.plotter.add_slider_widget(
            self._on_time_slider_changed,
            [0, self.num_steps-1],
            value=0,
            title="time",
            pointa=(0.66, 0.95),
            pointb=(1.0, 0.95),
            style='modern',
            fmt="",
            interaction_event='always',
            tube_width=0.02,
            slider_width=0.02,
            title_height=0.02,
            title_color=self.palette['text'],
            color=self.palette['accent'],
        )
        # 播放速度滑块
        self.speed_slider = self.plotter.add_slider_widget(
            self._on_speed_changed,
            [0.1, 2.0],
            value=1.0,
            title="animation speed",
            pointa=(0.8, 0.9),
            pointb=(1.0, 0.9),
            style='modern',
            fmt="",
            tube_width=0.01,
            slider_width=0.01,
            title_height=0.02,
            title_color=self.palette['text'],
            color=self.palette['accent_dark'],
        )
        # 注意：底部按钮单独在 _setup_bottom_buttons 中统一布局

    def _setup_display_controls(self):
        """设置显示选项UI"""
        win_w, win_h = self.plotter.window_size
        self.plotter.add_text("边框", position=(int(win_w*0.82), int(win_h*0.82)), font_size=9, color='#e5e7eb',
            **self.text_kwargs)
        self.edge_toggle = self.plotter.add_checkbox_button_widget(
            self._on_edges_toggled, value=self.show_edges,
            position=(int(win_w*0.80), int(win_h*0.815)), size=20, border_size=2
        )
        self.plotter.add_text("自动颜色范围", position=(int(win_w*0.82), int(win_h*0.76)), font_size=9, color='#e5e7eb',**self.text_kwargs)
        self.clim_toggle = self.plotter.add_checkbox_button_widget(
            self._on_clim_toggled, value=self.auto_clim,
            position=(int(win_w*0.80), int(win_h*0.755)), size=20, border_size=2
        )

    def _setup_bottom_buttons(self):
        """底部中部按钮：第一帧 | 上一帧 | 播放 | 下一帧 | 最后一帧 | 重置相机 | 帮助"""
        win_w, win_h = self.plotter.window_size
        btn_size = 40
        gap = 10
        base = (btn_size + gap)
        center_x = int(win_w // 2)
        y = int(win_h * 0.03)  # 底部略上方，避免紧贴边缘

        def pos_x(offset_idx: int) -> int:
            # 以底部中心为锚点，七个按钮对称分布
            return center_x + offset_idx * base - btn_size // 2

        # 第一帧（-3）
        x_first = pos_x(-3)
        self._btn_first = self.plotter.add_checkbox_button_widget(
            lambda state: self._on_first_click(),
            value=False,
            position=(x_first, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_on'],
            background_color=self.palette['button_on']
        )
        self.plotter.add_text("First", position=(x_first, y + btn_size*1//5), font_size=btn_size//5, color='white', **self.text_kwargs)

        # 上一帧（-2）
        x_prev = pos_x(-2)
        self._btn_prev = self.plotter.add_checkbox_button_widget(
            lambda state: self._on_prev_click(),
            value=False,
            position=(x_prev, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_on'],
            background_color=self.palette['button_on']
        )
        self.plotter.add_text("←", position=(x_prev, y), font_size=btn_size//2, color='#e5e7eb', **self.text_kwargs)

        # 播放/暂停（-1）
        x_play = pos_x(-1)
        self.play_button = self.plotter.add_checkbox_button_widget(
            self._on_play_toggled,
            value=False,
            position=(x_play, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_off'],
            background_color=self.palette['button_off']
        )
        self.plotter.add_text(" Play", position=(x_play,y+btn_size//5), font_size=btn_size//5, color='white', **self.text_kwargs)

        # 下一帧（0）
        x_next = pos_x(0)
        self._btn_next = self.plotter.add_checkbox_button_widget(
            lambda state: self._on_next_click(),
            value=False,
            position=(x_next, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_on'],
            background_color=self.palette['button_on']
        )
        self.plotter.add_text("→", position=(x_next, y), font_size=btn_size//2, color='white', **self.text_kwargs)

        # 最后一帧（+1）
        x_last = pos_x(1)
        self._btn_last = self.plotter.add_checkbox_button_widget(
            lambda state: self._on_last_click(),
            value=False,
            position=(x_last, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_on'],
            background_color=self.palette['button_on']
        )
        self.plotter.add_text("Last", position=(x_last, y + btn_size*1//5), font_size=btn_size//5, color='white', **self.text_kwargs)

        # 重置相机（+2）
        x_reset = pos_x(2)
        self._btn_reset = self.plotter.add_checkbox_button_widget(
            lambda state: self._on_reset_click(),
            value=False,
            position=(x_reset, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_on'],
            background_color=self.palette['button_on']
        )
        self.plotter.add_text("Reset", position=(x_reset , y+btn_size*1//5), font_size=btn_size//5, color='white', **self.text_kwargs)

        # 帮助（+3）
        x_help = pos_x(3)
        self._btn_help = self.plotter.add_checkbox_button_widget(
            lambda state: self._on_help_click(),
            value=False,
            position=(x_help, y),
            size=btn_size,
            border_size=0,
            color_on=self.palette['button_on'],
            color_off=self.palette['button_off'],
            background_color=self.palette['button_off']
        )
        self.plotter.add_text("？", position=(x_help , y), font_size=btn_size//2, color=self.palette['text'], **self.text_kwargs)

    
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
        
        # D键：切换下一场
        self.plotter.add_key_event("d", self._cycle_next_field)
        
        # A键：切换上一场
        self.plotter.add_key_event("a", self._cycle_prev_field)
        
        # R键：重置视图
        self.plotter.add_key_event("r", self._reset_view)

        # H键：切换帮助文本
        self.plotter.add_key_event("h", self._toggle_help)
    
    def _relayout_ui(self):
        # 移除旧控件，再按新尺寸重建
        try:
            self.plotter.clear_slider_widgets()
            self.plotter.clear_button_widgets()   # 如果版本不支持，可忽略
        except Exception:
            pass
        self.plotter.clear_actors()
        self._setup_ui()
        self.plotter.render()

    # ========== 回调函数 ==========
    
    def _on_time_slider_changed(self, value):
        """时间滑块回调：更新当前索引并刷新显示。"""
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
        try:
            self.play_button.GetRepresentation().SetState(1 if self.is_playing else 0)
        except Exception:
            pass
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
        new_speed = max(0.1, min(2.0, self.play_speed + delta))
        self.play_speed = new_speed
        self.speed_slider.GetRepresentation().SetValue(new_speed)
        print(f"播放速度: {self.play_speed:.1f}")
    

    def _cycle_next_field(self):
        """循环切换场"""
        if not self.available_fields:
            return
        current_index = self.available_fields.index(self.current_field) if self.current_field in self.available_fields else 0
        next_index = (current_index + 1) % len(self.available_fields)
        self.current_field = self.available_fields[next_index]
        print(f"选择场: {self.current_field}")
        self._display_current_step()
    
    def _cycle_prev_field(self):
        """循环切换场"""
        if not self.available_fields:
            return
        current_index = self.available_fields.index(self.current_field) if self.current_field in self.available_fields else 0
        prev_index = (current_index - 1+len(self.available_fields)) % len(self.available_fields)
        self.current_field = self.available_fields[prev_index]
        print(f"选择场: {self.current_field}")
        self._display_current_step()

    def _reset_view(self):
        """重置视图并自动选择 2D/3D 相机模式。"""
        # 相机根据维度调整（仅首次）
        bounds = self.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
        z_thickness = bounds[5] - bounds[4]
        xy_scale = max(bounds[1]-bounds[0], bounds[3]-bounds[2], 1e-9)
        is_2d = (z_thickness < 1e-6) or (z_thickness < 0.01 * xy_scale)
        if is_2d:
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()
        else:
            self.plotter.disable_parallel_projection()
            self.plotter.view_vector((-1,-1,1))
            # self.plotter.view_isometric(negative=[True,True,False])

        self.plotter.reset_camera(bounds =bounds)
        self.plotter.render()

    def _on_prev_click(self):
        self._prev_frame()
        # 恢复按钮为未选中
        try:
            self._btn_prev.GetRepresentation().SetState(0)
        except Exception:
            pass

    def _on_first_click(self):
        self.current_idx = 0
        try:
            self.time_slider.GetRepresentation().SetValue(self.current_idx)
        except Exception:
            pass
        self._display_current_step()
        try:
            self._btn_first.GetRepresentation().SetState(0)
        except Exception:
            pass

    def _on_next_click(self):
        self._next_frame()
        try:
            self._btn_next.GetRepresentation().SetState(0)
        except Exception:
            pass

    def _on_last_click(self):
        self.current_idx = max(0, self.num_steps - 1)
        try:
            self.time_slider.GetRepresentation().SetValue(self.current_idx)
        except Exception:
            pass
        self._display_current_step()
        try:
            self._btn_last.GetRepresentation().SetState(0)
        except Exception:
            pass

    def _on_reset_click(self):
        self._reset_view()
        try:
            self._btn_reset.GetRepresentation().SetState(0)
        except Exception:
            pass

    def _on_help_click(self):
        self._toggle_help()
        try:
            self._btn_help.GetRepresentation().SetState(1 if self._help_detailed else 0)
        except Exception:
            pass

    def _toggle_help(self):
        self._help_detailed = not self._help_detailed
        self._update_help_text()

    def _update_help_text(self):
        # 左上角：帮助提示（简洁/详细）
        if self._help_text_actor is not None:
            try:
                self.plotter.remove_actor(self._help_text_actor, reset_camera=False)
            except Exception:
                pass
            self._help_text_actor = None
        if self._help_detailed:
            text = (
                "快捷键:\n"
                "H: 切换帮助\n"
                "Space: 播放/暂停\n"
                "←/→: 上一帧/下一帧\n"
                "+/-: 调整播放速度\n"
                "F: 切换场\n"
                "R: 重置相机"
            )
        else:
            text = ""
        # 顶部左上锚点，向下对齐，避免溢出
        self._help_text_actor = self.plotter.add_text(text, position='upper_left', font_size=10, color='#e5e7eb', **self.text_kwargs)
        try:
            tp = self._help_text_actor.GetTextProperty()
            tp.SetVerticalJustificationToTop()
            tp.SetJustificationToLeft()
        except Exception:
            pass
    
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
            try:
                self.play_button.GetRepresentation().SetState(0)
            except Exception:
                pass


    def _display_current_step(self):
        """显示当前时间步：懒加载网格、应用场与配色、更新标尺与文本。

        通过延迟渲染与 actor 置换减少闪烁；颜色条位置与样式由
        `self._cbar_conf` 与主题控制。
        """
        win_w,win_h = self.plotter.window_size 
        # 不立即清场景，先准备新 actor，减少闪烁
        
        mesh = self._load_mesh(self.current_idx)
        if mesh is None or mesh.n_points == 0:
            print(f"警告: 时间步 {self.current_idx} 为空网格")
            return

        scalars = None
        if self.current_field and self.current_field in mesh.array_names:
            scalars = self.current_field
        elif mesh.array_names:
            for field in mesh.array_names:
                if field not in ['Normals', 'TCoords']:
                    scalars = field
                    break

        clim = None
        if scalars and not self.auto_clim:
            data = mesh[scalars]
            if data is not None and len(data) > 0:
                clim = [np.min(data), np.max(data)]
        
        # 先移除现有标尺，确保新场/新范围能重建颜色条
        try:
            if getattr(self.plotter, "scalar_bars", None):
                for k in list(self.plotter.scalar_bars._scalar_bar_actors.keys()):
                    try:
                        self.plotter.remove_scalar_bar(title=k, render=False)
                    except Exception:
                        pass
        except Exception:
            pass

        # 添加网格 + 颜色条（右下角）；先不渲染
        new_actor = self.plotter.add_mesh(
            mesh,
            scalars=scalars,
            show_edges=self.show_edges and mesh.n_cells < 20000,
            cmap=self.current_cmap,
            clim=clim,
            scalar_bar_args={
                'title': scalars if scalars else 'Value',
                'vertical': True,
                'title_font_size': 20,
                'label_font_size': 20,
                'color': self.palette['text'],
                'shadow': False,
                'n_labels': 4,
                'position_x': self._cbar_conf['position_x'],
                'position_y': self._cbar_conf['position_y'],
                'width': self._cbar_conf['width'],
                'height': self._cbar_conf['height'],
            },
            lighting=True,
            smooth_shading=True,
            ambient=0.3,
            specular=0.4,
            specular_power=20,
            nan_color='#6b7280',
            render=False  # 延后统一渲染，避免闪屏
        )

        # 移除旧 mesh 后再赋值，减小闪烁
        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor, reset_camera=False, render=False)
            except Exception:
                pass
        self._mesh_actor = new_actor

        # 此处不再清理标尺，上一段已清空旧标尺，新标尺随 add_mesh 创建
        
        # 左下角：方向轴（固定一次）
        if not self._axes_widget_added:
            try:
                self.plotter.add_orientation_widget(pv.Axes(), viewport=(0.01, 0.01, 0.18, 0.18))
                self._axes_widget_added = True
            except Exception:
                # 回退到默认 add_axes（位置不可控）
                self.plotter.add_axes(line_width=2, cone_radius=0.3, shaft_length=0.8, tip_length=0.2, ambient=0.4, label_size=(0.25, 0.25))
                self._axes_widget_added = True

        # 顶部右侧：时间信息（time step / real time）
        if self._time_text_actor is not None:
            try:
                self.plotter.remove_actor(self._time_text_actor, reset_camera=False)
            except Exception:
                pass
            self._time_text_actor = None
        time_text = f"真实时间：{self.time_values[self.current_idx]:.6f}"
        # 顶部右上，向下对齐，避免溢出
        self._time_text_actor = self.plotter.add_text(time_text, position='upper_right', font_size=10, color='#e5e7eb', **self.text_kwargs)
        try:
            tp = self._time_text_actor.GetTextProperty()
            tp.SetVerticalJustificationToTop()
            tp.SetJustificationToRight()
        except Exception:
            pass

        # 颜色条上方：场名
        if self._field_text_actor is not None:
            try:
                self.plotter.remove_actor(self._field_text_actor, reset_camera=False)
            except Exception:
                pass
            self._field_text_actor = None
        cb = self._cbar_conf

        label_x = int(win_w * (cb['position_x'] + cb['width']/2.0) - 30)
        label_y = int(win_h * (cb['position_y'] + cb['height']) + 8)
        field_name = scalars if scalars else 'None'

        # 顶部中间：标题
        title = f"PVD Viewer - {self.name}"
        self.plotter.add_title(title, font_size=10, color='#67e8f9')
        self.bounds = mesh.bounds
        # self._reset_view()
        # 统一渲染一次，减少闪烁
        self.plotter.render()

    
    def _compute_window_size(self):
        """根据屏幕分辨率设定窗口大小 (80%)。"""
        try:
            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            return int(width * 0.8), int(height * 0.8)
        except Exception:
            return 1280, 900
    # ========= 新增：懒加载 + LRU 缓存 =========
    def _load_mesh(self, idx, silent=False):
        """按需加载并缓存 mesh（LRU），支持静默模式。"""
        if idx < 0 or idx >= len(self.vtk_files):
            if not silent:
                print(f"索引 {idx} 超出范围")
            return None
        # 缓存命中
        if idx in self.mesh_cache:
            mesh = self.mesh_cache.pop(idx)
            self.mesh_cache[idx] = mesh
            return mesh
        vtk_file = self.vtk_files[idx]
        try:
            mesh = pv.read(vtk_file)
            self.blocks[idx] = mesh
            # 维护 LRU
            self.mesh_cache[idx] = mesh
            if len(self.mesh_cache) > self.cache_size:
                self.mesh_cache.popitem(last=False)
            if not silent:
                print(f"  加载: {Path(vtk_file).name}")
            return mesh
        except Exception as e:
            if not silent:
                print(f"  [!] 加载失败 {vtk_file}: {e}")
            return None
    def show(self):
        """显示可视化窗口并打印基本操作说明。"""
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
