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
        preview_size = (self._scaled(390, 260), self._scaled(230, 170))
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
                self.tube_params = config["tube_params"]
                sample_rate = str(self.tube_params["sample_rate"])
                self.sample_rate_value.setCurrentText(sample_rate)
                self.sample_area_value.setText(str(self.tube_params["sample_area_mm2"]))
                self.incident_tube_radius_value.setText(str(self.tube_params["incident_tube_radius_mm"]))
                self.rear_mic_sensitivity_value.setText(str(self.tube_params["rear_mic_sensitivity_v_per_pa"]))
                self.back_cavity_radius_value.setText(str(self.tube_params["back_cavity_radius_mm"]))
                self.front_mic_distance_value.setText(str(self.tube_params["front_mic_distance_mm"]))
                self.back_cavity_depth_value.setText(str(self.tube_params["back_cavity_depth_mm"]))
                self.tube_temperature_value.setText(str(self.tube_params["tube_temperature"]))

        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取配置失败：{e}")

    def save(self):
        config_path = utils.get_config_path("imp_tube_params_setting.json")
        try:
            sample_rate = int(self.sample_rate_value.currentText())
            if sample_rate not in (44100, 48000):
                raise ValueError("采样率只能选择 44100 Hz 或 48000 Hz。")

            tube_params = {
                "sample_rate": sample_rate,
                "nfft": sample_rate,
                "sample_area_mm2": float(self.sample_area_value.text()),
                "incident_tube_radius_mm": float(self.incident_tube_radius_value.text()),
                "rear_mic_sensitivity_v_per_pa": float(self.rear_mic_sensitivity_value.text()),
                "back_cavity_radius_mm": float(self.back_cavity_radius_value.text()),
                "front_mic_distance_mm": float(self.front_mic_distance_value.text()),
                "back_cavity_depth_mm": float(self.back_cavity_depth_value.text()),
                "tube_temperature": float(self.tube_temperature_value.text()),
                "atmospheric_pressure_kpa": float(self.tube_params["atmospheric_pressure_kpa"]),
            }
            for key, value in tube_params.items():
                if key != "tube_temperature" and value <= 0:
                    raise ValueError(f"{key} 必须大于 0。")
            config = {
                "tube_params": tube_params
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.tube_params = tube_params
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
