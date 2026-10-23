import os
import json

import numpy as np
import pyqtgraph
import sounddevice as sd
from scipy.io import wavfile
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QDialog, QMessageBox, QVBoxLayout
from PyQt5.uic import loadUi

from control.log_manager import LogManager
from control.utils import utils
from custom.customSignals import sign
from custom.running_consts import base_dir


class OutputSignalSetInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.signal_info = None
        # 缩放比例
        self._ui_scale = 1.0
        self.logger = LogManager.set_log_handler("输出信号设置")
        self.init_ui()
        self.init_fun()
        self.graph_signal()

    def init_ui(self):
        ui_file = QFile(":ui/output_signal_setting.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 output_signal_setting.ui")
            raise FileNotFoundError("未找到资源文件 output_signal_setting.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("输出信号设置")
        # 最小化|最大化
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint| Qt.WindowMaximizeButtonHint)
        # 按屏幕可用区域缩小窗口
        utils.resize_by_ui_with_screen(self)
        self.init_view()
        self.init_config()
        self.show()
        self.logger.info("打开输出信号设置界面")

    def init_view(self):
        self.plot = pyqtgraph.PlotWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.ui.frame.setLayout(layout)
        self.ui.plot.setBackground('white')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude', units='V')
        self.plot.getAxis('bottom').setTextPen("black")
        self.plot.getAxis('left').setTextPen("black")
        self.plot.showGrid(x=True, y=True, alpha=0.25)

    def init_fun(self):
        self.update_view.clicked.connect(self.graph_signal)
        self.save_params.clicked.connect(self.save_params_to_json)
        self.save_wav_button.clicked.connect(self.save_wav)
        self.play_button.clicked.connect(self.play_clicked)
        sign.play_audio_sign.connect(self.play_audio,Qt.AutoConnection)
        sign.error_message_signal.connect(utils.show_error_message)

    def init_config(self):
        config_path = utils.get_config_path("output_signal_setting.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.signal_info = config.get("signal_info", {})
            self.signal_time_val = float(self.signal_info.get("signal_time", 10))
            self.hz_down_val = float(self.signal_info.get("hz_down", 1))
            self.hz_up_val = float(self.signal_info.get("hz_up", 10))
            self.signal_amplitude_val = float(self.signal_info.get("signal_amplitude", 1))

            self.signal_time.setText(str(self.signal_time_val))
            self.hz_down.setText(str(self.hz_down_val))
            self.hz_up.setText(str(self.hz_up_val))
            self.signal_amplitude.setText(str(self.signal_amplitude_val))
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取配置失败：{e}")

    def graph_signal(self):
        try:
            duration = float(self.signal_time.text())
            f_start = float(self.hz_down.text())
            f_stop = float(self.hz_up.text())
            amplitude = float(self.signal_amplitude.text())
        except ValueError:
            QMessageBox.warning(self, "格式错误", "请输入合法数字")
            return
        x, y = utils.generate_chirp(duration, f_start, f_stop, amplitude)

        self.plot.clear()
        self.plot.plot(x, y, pen='blue')
        self.plot.setXRange(0, duration)
        self.plot.setYRange(-1.1 * amplitude, 1.1 * amplitude) # 留边界

    def save_params_to_json(self):
        config_path = utils.get_config_path("output_signal_setting.json")
        try:
            existing_path = None
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
                    existing_path = existing_config.get("signal_path", None)
            self.signal_info = {
                "signal_time": float(self.signal_time.text()),
                "hz_down": float(self.hz_down.text()),
                "hz_up": float(self.hz_up.text()),
                "signal_amplitude": float(self.signal_amplitude.text())
            }
            config = {
                "signal_info": self.signal_info,
                "signal_path": existing_path  # 保留原路径
            }
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"配置已保存到 {config_path}")
            QMessageBox.information(self, "成功", "参数保存成功！")
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            QMessageBox.critical(self, "错误", f"保存配置失败：{e}")

    def play_audio(self):
        try:
            # 获取设备采样率
            samplerate = utils.get_device_info()
            info = {
                "signal_time": float(self.signal_time.text()),
                "hz_down": float(self.hz_down.text()),
                "hz_up": float(self.hz_up.text()),
                "signal_amplitude": float(self.signal_amplitude.text())
            }
            result, data, amplitude, msg = utils.generate_calibrated_signal(info, samplerate)
            if result is False:
                raise ValueError(f"{msg}")
            sd.play(data, samplerate)
            sd.wait()

        except Exception as e:
            self.logger.error(f"播放音频失败: {e}")
            sign.error_message_signal.emit(f"播放音频失败：{e}", self)

    def play_clicked(self):
        sign.play_audio_sign.emit()

    def save_wav(self):
        try:
            # 获取设备采样率
            samplerate = utils.get_device_info()
            self.signal_info = {
                "signal_time": float(self.signal_time.text()),
                "hz_down": float(self.hz_down.text()),
                "hz_up": float(self.hz_up.text()),
                "signal_amplitude": float(self.signal_amplitude.text())
            }
            filename = (f"chirp_{self.signal_info['signal_time']:.2f}s_{self.signal_info['hz_down']:.1f}-"
                        f"{self.signal_info['hz_up']:.1f}Hz_{self.signal_info['signal_amplitude']:.2f}V.wav")
            result, data, amplitude, msg = utils.generate_calibrated_signal(self.signal_info, samplerate)
            if result is False:
                raise ValueError(f"{msg}")
            save_dir = os.path.join(base_dir, "audio_data")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, filename)
            save_path = save_path.replace("\\", "/")
            # 保存 WAV 文件
            data = data.astype('float32')
            wavfile.write(save_path, samplerate, data)
            self.logger.info(f"音频已保存至：{save_path}")
            QMessageBox.information(self, "成功", f"音频保存成功：\n{save_path}")

        except Exception as e:
            self.logger.error(f"保存音频失败: {e}")
            QMessageBox.critical(self, "错误", f"保存音频失败：{e}")

    def closeEvent(self, event):
        if self.parent:
            self.parent.output_window = None
        sign.play_audio_sign.disconnect(self.play_audio)
        sign.error_message_signal.disconnect(utils.show_error_message)
        self.logger.info("关闭输出信号设置界面")
        event.accept()

    # @pyqtSlot(str)
    # def show_error_message(self, msg: str) -> None:
    #     """在主线程弹出错误提示框。"""
    #     QMessageBox.critical(self, "错误", msg)
