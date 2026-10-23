import os
# 必须在导入 sounddevice 之前设置这个环境变量
os.environ["SD_ENABLE_ASIO"] = "1"
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from control.utils import utils
from control.window import MainWindow


if __name__ == '__main__':
    # 必须在 QApplication 创建前启用，否则 Windows 会在 125%/150% 缩放下
    # 对整个窗口做位图缩放，导致文字、曲线和示意图发虚。
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    utils.copy_all_configs_to_base_dir()
    app = QApplication(sys.argv)
    app.setApplicationName("小孔膜片阻抗管测试系统 V2.0")
    screen_metrics = utils.get_screen_metrics()
    print(
        "[屏幕信息] "
        f"{screen_metrics['name']} "
        f"分辨率={screen_metrics['width']}x{screen_metrics['height']}, "
        f"可用区域={screen_metrics['available_width']}x{screen_metrics['available_height']}, "
        f"DPI={screen_metrics['logical_dpi']}, "
        f"缩放系数={screen_metrics['device_pixel_ratio']}"
    )
    utils.init_selected_devide()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
