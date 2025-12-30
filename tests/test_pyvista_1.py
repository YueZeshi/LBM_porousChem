import pyvista as pv
import numpy as np

p = pv.Plotter(window_size=(800, 600))

# 添加背景网格作为参考
mesh = pv.Sphere()
p.add_mesh(mesh)

# 添加各种widget测试
print("添加widgets...")

# 1. 滑块（应该可见）
try:
    p.add_slider_widget(
        lambda x: print(f"Slider: {x}"),
        [0, 10],
        value=5,
        title="Test Slider",
        pointa=(0.1, 0.1),
        pointb=(0.4, 0.1)
    )
    print("✓ 滑块添加成功")
except Exception as e:
    print(f"✗ 滑块失败: {e}")

# 2. Checkbox按钮
try:
    p.add_checkbox_button_widget(
        lambda state: print(f"Checkbox: {state}"),
        value=False,
        position=(0.5, 0.9),
        size=40,
        color_on='red',
        color_off = "black",
        background_color='white'
    )
    print("✓ Checkbox添加成功")
except Exception as e:
    print(f"✗ Checkbox失败: {e}")

# 3. 文本
try:
    p.add_text(
        "Test Text",
        position=(0.5, 0.5),
        font_size=20,
        color='yellow'
    )
    print("✓ 文本添加成功")
except Exception as e:
    print(f"✗ 文本失败: {e}")

print("\n显示窗口...")
p.show()