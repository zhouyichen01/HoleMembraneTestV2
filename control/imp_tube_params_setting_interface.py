import json

from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUi

from control.log_manager import LogManager
from control.utils import utils

class ImptubeParamsSetInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        # 缩放比例
        self._ui_scale = 1.0
        self.logger = LogManager.set_log_handler("阻抗管参数设置")
        self.init_ui()
        self.init_fun()

    def init_ui(self):
        ui_file = QFile(":ui/imp_tube_params_setting.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 imp_tube_params_setting.ui")
            raise FileNotFoundError("未找到资源文件 imp_tube_params_setting.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("阻抗管参数设置")
        # 最小化|最大化
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        # 按屏幕可用区域缩小窗口
        utils.resize_by_ui_with_screen(self)
        self.init_images()
        self.init_config()
        self.show()
        self.logger.info("打开阻抗管参数设置界面")

    def _scaled(self, value, minimum):
        scaled_value = utils.scale_ui_value(self, value, minimum)
        return scaled_value

    def init_images(self):
        # 示意图缩略图原本是固定尺寸，小屏幕下跟随窗口比例缩小。
        preview_size = (self._scaled(390, 260), self._scaled(282, 188))
        utils.set_clickable_graphics_image(
            parent=self,
            view=self.graphicsView,
            image_path=":/images/2mic薄层_画板2026_hd.png",
            preview_size=preview_size,
            title="参数设置图片预览",
        )

    def init_fun(self):
        self.save_button.clicked.connect(self.save)

    def init_config(self):
        config_path = utils.get_config_path("imp_tube_params_setting.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.tube_params = config.get("tube_params", {})
                self.tube_temperature_value.setText(str(self.tube_params.get("tube_temperature", "")))
                self.sample_area_value.setText(str(self.tube_params.get("s_sample_mm2", "")))
                self.v_backing_cc_value.setText(str(self.tube_params.get("v_backing_cc", "")))

        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取配置失败：{e}")

    def save(self):
        config_path = utils.get_config_path("imp_tube_params_setting.json")
        try:
            s_sample_mm2 = float(self.sample_area_value.text())
            config = {
                "tube_params": {
                    "tube_temperature": float(self.tube_temperature_value.text()),
                    "s_sample_mm2": s_sample_mm2,
                    "v_backing_cc": float(self.v_backing_cc_value.text())
                }
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            if self.parent:
                self.parent.tube_params = config["tube_params"]
            QMessageBox.information(self, "成功", "参数保存成功")
            self.logger.info("参数保存成功")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败：{e}")
            self.logger.error(f"保存失败：{e}")

    def closeEvent(self, event):
        if self.parent:
            self.parent.imptube_window = None
        self.logger.info("关闭阻抗管参数设置界面")
        event.accept()
