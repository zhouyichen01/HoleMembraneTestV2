import json
import os

import numpy as np
import sounddevice as sd

from PyQt5.QtCore import QFile, Qt, QTimer
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets

from control.log_manager import LogManager
from control.utils import utils
from control.utils.audio_session_manager import AudioSessionManager
from control.utils.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from control.utils.streaming_audio_processor import DuplexStreamingPlayRec


class MicoutputVoltageInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.output_voltage_value = []
        self.current_count = 1
        self.calibration_param = None
        self.play_flag = False
        self.countdown = 10
        self.ui = None
        self.v2pa_factor = None
        self.sensitivity = None
        self.stream_instance = None
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._handle_input_calibration_stream)
        self._input_channel_index = None
        self._input_vp = None
        self.logger = LogManager.set_log_handler("输入/输出校准")
        self.init_ui()
        self.init_fun()
        self.get_calibration_param()

    def init_ui(self):
        ui_file = QFile(":ui/output_voltage_cal.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 output_voltage_cal.ui")
            raise FileNotFoundError("未找到资源文件 output_voltage_cal.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("输入/输出校准")
        self.ui.tabWidget.setTabText(0, "输出校准")
        self.ui.tabWidget.setTabText(1, "输入校准")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint| Qt.WindowMaximizeButtonHint)
        self.show()
        self.logger.info("打开输入/输出校准")

    def init_fun(self):
        # 输出界面初始化
        self.calibration_nums_box.setValue(1)
        self.yincang.setStyleSheet("background-color: transparent; border: none; color: transparent;")
        self.countdown_label.setStyleSheet("background-color: transparent; border: none; color: transparent;")
        self.calibration_nums_box.setRange(1, 20)
        self.reset_button.clicked.connect(self.clicked_reset_button)
        self.play_button.clicked.connect(self.play_btn_clicked)
        self.save_button.clicked.connect(self.save_btn_clicked)
        self.calibration_nums_box.valueChanged.connect(self.get_calibration_param)
        self.test_button.clicked.connect(self.test_calibration)
        self.cal_button.clicked.connect(self.clicked_calibration_button)
        # 输入界面初始化
        self.standard_spl_i.setChecked(True)
        self.init_channel_combobox()
        self.save_mic_button.clicked.connect(self.save_mic_deviation_value)

    def get_calibration_param(self):
        calibration_nums = self.calibration_nums_box.value()
        output_voltage = self.output_voltage_value
        if calibration_nums == 1:
            amplitude_list = np.array([0.95])
        else:
            amplitude_list = np.linspace(0.15, 0.95, calibration_nums)
        self.calibration_param = {
            "calibration_nums": calibration_nums,
            "output_voltage": output_voltage,
            "amplitude_list": amplitude_list
        }

    def play_btn_clicked(self):
        stimulus_dict = self.create_signal()
        if not self.play_flag:
            self.play_flag = True
            self.calibration_nums_box.setEnabled(False)
            self.play_button.setEnabled(False)
            # self.countdown_label.setText(f"<span style='color: black;'>第{self.current_count}次播放中 </span>")
            QtWidgets.QApplication.processEvents()
            play_result, msg = self.sd_play(stimulus_dict)
            if play_result:
                self.logger.info(f"第{self.current_count}次播放成功")
                self.play_button.setText("停止")
                self.save_button.setEnabled(True)
                self.play_flag = False
            else:
                self.logger.error(f"第{self.current_count}次播放失败,{msg}")
                self.reset_output_clicked()

    def create_signal(self):
        samplerate = utils.get_device_info()
        config_path = utils.get_config_path("output_signal_setting.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                signal_info = config.get("signal_info", {})
                signal_info['signal_amplitude'] = 1  # 归一化到 1
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取配置失败：{e}")

        _, data = utils.generate_chirp_wrapper(signal_info, samplerate)
        stimulus_dict = {"data": data,
                         "sr": samplerate,
                         "amplitude": self.calibration_param["amplitude_list"][self.current_count - 1],
                         }
        return stimulus_dict

    def create_current_count(self):
        if self.current_count >= self.calibration_param["calibration_nums"]:
            self.save_button.setText("完成")
            self.save_button.setEnabled(False)
            self.play_button.setEnabled(False)
        else:
            self.current_count += 1
            self.play_button.setText("播放")
            self.play_button.setEnabled(True)
            self.save_button.setEnabled(False)

    def test_calibration(self):
        try:
            params_voltage = self.target_voltage_box.value()
            samplerate = utils.get_device_info()
            config_path = utils.get_config_path("output_signal_setting.json")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    signal_info = config.get("signal_info", {})
                    signal_info['signal_amplitude'] = params_voltage
            except Exception as e:
                raise ValueError(f"{e}")
            result, data, amplitude, msg = utils.generate_calibrated_signal(signal_info, samplerate)
            if result is False:
                raise ValueError(f"{msg}")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            self.logger.error(f"测试失败: {e}")
            QMessageBox.critical(self, "错误", f"测试失败：{e}")

    @staticmethod
    def sd_play(stimulus_params):
        try:
            data = stimulus_params.get("data") * stimulus_params.get("amplitude")
            sr = stimulus_params.get("sr")
            blocking = stimulus_params.get("blocking", True)
            sd.play(data, samplerate=sr, blocking=blocking)
            return True, None
        except Exception as e:
            err_msg = "Failed to play audio. [%s]" % (str(e))
            return False, err_msg

    def save_btn_clicked(self):
        self.output_voltage_value.append(self.output_voltage_box.value())
        self.output_voltage_box.setValue(0)
        self.create_current_count()
        self.play_label.setText(f"第{self.current_count}次 ")

    def clicked_calibration_button(self):
        current_tab_index = self.tabWidget.currentIndex()
        if current_tab_index == 0:
            self.output_calibration()
        elif current_tab_index == 1:
            self.record_state.setText("录制中")
            self.record_state.setStyleSheet("color: green;")
            QtWidgets.QApplication.processEvents()
            self.input_calibration()

    def clicked_reset_button(self):
        current_tab_index = self.tabWidget.currentIndex()
        if current_tab_index == 0:
            self.reset_output_clicked()
        elif current_tab_index == 1:
            self.reset_intput_clicked()

    def output_calibration(self):
        """
        输出电压校准
        通过校准数据与给定的参数进行拟合，检查是否校准成功。
        """
        scm = SoundcardCalibrationManager()
        if len(self.output_voltage_value) != self.calibration_param["calibration_nums"]:
            QMessageBox.critical(self, "错误", "所保存的电压不符合校准次数的要求")
            self.logger.error("所保存的电压不符合校准次数的要求")
        else:
            if self.calibration_param["calibration_nums"] == 1:
                scm.add_data(0, 0, validation=False)
            for amplitude, voltage in zip(self.calibration_param["amplitude_list"], self.output_voltage_value):
                scm.add_data(amplitude, voltage)
            fit_code, msg = scm.fit()
            if fit_code == True:
                QMessageBox.information(self, "成功", "校准成功！")
                self.logger.info(f"校准成功！保存的电压列表:{self.output_voltage_value}")
            else:
                QMessageBox.critical(self, "错误", f"校准失败： {msg}")
                self.logger.error(f"校准失败：{msg}, 保存的电压列表:{self.output_voltage_value}")

    def reset_output_clicked(self):
        self.output_voltage_value.clear()
        self.output_voltage_box.setValue(0)
        self.target_voltage_box.setValue(0)
        self.calibration_nums_box.setValue(1)
        self.current_count = 1
        self.countdown = 10
        self.play_label.setText(f"第{self.current_count}次 ")
        self.play_button.setText("播放")
        self.play_flag = False
        self.calibration_nums_box.setEnabled(True)
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(False)

#=========================================================================================
    def init_channel_combobox(self):
        """
        初始化channel_number下拉框
        """
        try:
            config_path = utils.get_config_path("basic_params.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError("配置文件不存在")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                count = config["basic_params"]["input_channel_count"]
            # 动态填充下拉框：In1…InN
            self.ui.channel_number.clear()
            for i in range(1, count  + 1):
                self.ui.channel_number.addItem(f"In{i}", i)
            self.ui.channel_number.setCurrentIndex(0)

        except Exception as e:
            self.logger.warning(f"初始化下拉框失败: {e}")


    def input_calibration(self):
        """
        启动输入电压校准的流式录音。
        保持“录制中”状态，不显示或更新计时；录音结束后由定时器收尾。
        """
        self.v2pa_factor = None
        self.sensitivity = None
        samplerate = utils.get_device_info()
        res, content = utils.get_config_content("basic_params.json")
        if res:
            channels_counts = content["basic_params"]["input_channel_count"]
        else:
            QMessageBox.critical(self, "错误", f"校准失败：basic_params.json导入错误")
            self.logger.error(f"校准失败：basic_params.json导入错误")
            self.record_state.setText("待录制")
            self.record_state.setStyleSheet("color: rgb(255, 170, 0);")
            return

        # 获取下拉框通道数据
        channel_number_text = self.channel_number.currentText()
        res, config = utils.get_config_content("mic_calibration.json")
        if res:
            channel_index = config[channel_number_text]["binding_index"]
        else:
            QMessageBox.critical(self, "错误", f"校准失败：mic_calibration.json导入错误")
            self.logger.error(f"校准失败：mic_calibration.json导入错误")
            self.record_state.setText("待录制")
            self.record_state.setStyleSheet("color: rgb(255, 170, 0);")
            return
        if not AudioSessionManager.acquire(self):
            QMessageBox.warning(self, "提示", "音频设备正在被其它界面占用，请先停止/关闭其它录音功能。")
            self._set_input_record_idle()
            return

        try:
            input_device = content["basic_params"]["input_selected_device_id"]
            output_device = content["basic_params"]["output_selected_device_id"]
            input_channels = sd.query_devices(input_device, "input")["max_input_channels"]
            output_channels = sd.query_devices(output_device, "output")["max_output_channels"]
            if channels_counts > input_channels:
                raise ValueError(f"输入设备仅支持 {input_channels} 个通道，配置需要 {channels_counts} 个通道")
            if output_channels < 1:
                raise ValueError("输出设备不可用，无法启动音频流")

            self._input_channel_index = channel_index
            self._input_vp = content["basic_params"]["Vp"]
            duration = 10
            silence = np.zeros((duration * samplerate, int(output_channels)), dtype=np.float32)
            self.cal_button.setEnabled(False)
            self.stream_instance = DuplexStreamingPlayRec()
            self.stream_instance.start(
                stimulus_data=silence,
                sample_rate=samplerate,
                input_device=input_device,
                output_device=output_device,
                input_channels=int(channels_counts),
                output_channels=int(output_channels),
                duration=duration,
                blocksize=2048,
            )
            self.stream_timer.start(50)
            self.logger.info("开始输入校准流式录音")
        except Exception as e:
            self.logger.exception("输入校准录音启动失败")
            QMessageBox.critical(self, "录音错误", f"录音失败：{e}")
            self._set_input_record_idle()
            AudioSessionManager.release(self)

    def _handle_input_calibration_stream(self):
        """持续取出回调数据；不更新计时文本，录制结束后再计算校准结果。"""
        if self.stream_instance is None:
            return

        self.stream_instance.process_queue()
        if self.stream_instance.is_recording:
            return

        self.stream_timer.stop()
        stream = self.stream_instance
        self.stream_instance = None
        recording = stream.get_recorded_data()
        error = stream.error
        stream.stop()

        try:
            if error:
                raise RuntimeError(error)
            if recording.shape[0] == 0:
                raise ValueError("未采集到录音数据")

            recorded_data = recording[:, self._input_channel_index]
            # 延续原有逻辑：丢弃启动后的第 1 秒，再从中段取稳定数据计算。
            recorded_data = recorded_data[int(stream.sample_rate):]
            self.average_value, dbfs = self.calculate_average_spl(recorded_data, self._input_vp)
            v2pa_factor, sensitivity = self.calculate_deviation(self.average_value, dbfs)
            self._finish_input_calibration(v2pa_factor, sensitivity)
        except Exception as e:
            self.logger.exception("输入校准录音处理失败")
            QMessageBox.critical(self, "录音错误", f"录音失败：{e}")
        finally:
            self._set_input_record_idle()
            AudioSessionManager.release(self)

    def _finish_input_calibration(self, v2pa_factor, sensitivity):
        self.deviation_lineedit.setText(f"{sensitivity} V/Pa")
        if not np.isfinite(v2pa_factor) or v2pa_factor <= 0:
            QMessageBox.critical(self, "错误", f"校准失败：偏差值为{v2pa_factor}")
            self.logger.error(f"校准失败：偏差值为{v2pa_factor}")
        elif not np.isfinite(sensitivity) or sensitivity <= 0:
            QMessageBox.critical(self, "错误", f"校准失败：灵敏度为{sensitivity} V/Pa")
            self.logger.error(f"校准失败：灵敏度为{sensitivity} V/Pa")
        else:
            self.v2pa_factor, self.sensitivity = v2pa_factor, sensitivity
            QMessageBox.information(self, "成功", "校准成功！请保存数据！")
            self.logger.info(f"校准成功！")

    def _set_input_record_idle(self):
        self.record_state.setText("待录制")
        self.record_state.setStyleSheet("color: rgb(255, 170, 0);")
        self.cal_button.setEnabled(True)
        QtWidgets.QApplication.processEvents()

    def calculate_average_spl(self, recorded_data, vp):
        """
        保留原函数名：这里返回校准录音的 RMS，用于后续计算 V/Pa。
        """
        step = len(recorded_data) // 3
        if step == 0:
            raise ValueError("录音数据过短")
        spl_smooth, dbfs = AudioThdFrequencyResponseAnalysis().spl_calculation(
            recorded_data, method="rms", window_size=1201, vp=vp)
        spl_smooth_mid = len(spl_smooth) // 2
        spl_smooth_start = spl_smooth_mid - step
        spl_smooth_end = spl_smooth_mid + step
        spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
        return np.mean(spl_sample), dbfs

    def calculate_deviation(self, average_value, dbfs):
        """
        保留原函数名：根据录音 RMS 和标准声压计算灵敏度，单位 V/Pa。
        """
        if self.standard_spl_i.isChecked():
            deviation_value = round(94 - average_value, 1)
        else:
            deviation_value = round(114 - average_value, 1)
        sensitivity = 10 ** (dbfs / 20)
        v2pa_factor = 10 ** (deviation_value / 20)
        self.logger.info(
            f"校准偏差值：{deviation_value} dB，系数：{v2pa_factor:.3f}，"
            f"灵敏度：{sensitivity:.4f} V/Pa"
        )
        return round(v2pa_factor, 3), round(sensitivity, 4)

    def save_mic_deviation_value(self):
        """
        保存所选通道的偏差值和灵敏度到 json，灵敏度单位 V/Pa。
        """
        selected_channel = self.channel_number.currentText()
        if self.v2pa_factor is None or self.sensitivity is None:
            QMessageBox.warning(self, "警告", "请先校准Mic！")
            return

        channel_config_path = utils.get_config_path("mic_calibration.json")
        try:
            with open(channel_config_path, "r", encoding="utf-8") as f:
                channel_config = json.load(f)

            # 检查选中的麦克风是否在配置文件中
            if selected_channel in channel_config:
                channel_config[selected_channel]['deviation_value'] = str(self.v2pa_factor)
                channel_config[selected_channel]['sensitivity'] = str(self.sensitivity)
                with open(channel_config_path, "w", encoding="utf-8") as f:
                    json.dump(channel_config, f, ensure_ascii=False, indent=4)
                self.logger.info(f"成功保存 {selected_channel} 的偏差值和灵敏度")
                QMessageBox.information(self, "成功", f"保存成功！{selected_channel} 的偏差值和灵敏度已保存！")
            else:
                QMessageBox.warning(self, "错误", f"配置中未找到 {selected_channel} 的相关数据")
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            QMessageBox.warning(self, "错误", f"保存 {channel_config_path} 配置失败：{e}")

    def reset_intput_clicked(self):
        self._stop_input_calibration_stream()
        self.cal_button.setEnabled(True)
        self.deviation_lineedit.clear()
        self.v2pa_factor = None
        self.sensitivity = None

    def closeEvent(self, event):
        self._stop_input_calibration_stream()
        if self.parent:
            self.parent.output_voltage_window = None
        self.logger.info("关闭输入/输出校准")
        event.accept()

    def _stop_input_calibration_stream(self):
        if self.stream_timer.isActive():
            self.stream_timer.stop()
        if self.stream_instance is not None:
            self.stream_instance.stop()
            self.stream_instance = None
            AudioSessionManager.release(self)
        self._set_input_record_idle()

class SoundcardCalibrationManager(object):
    def __init__(self):
        self.amplitudes = []
        self.voltages = []
        self.logger = LogManager.set_log_handler("soundcard_core")

    def add_data(self, amplitude, voltage, validation=True):
        """
            添加幅度和电压数据。
            参数:
                amplitude: int 或 float
                    输入的幅度值。
                voltage: int 或 float
                    输入的电压值。
            返回:
                 包含状态码和提示信息的元组。
        """
        if validation:
            if not amplitude or not voltage:
                return False, "输入数据不能为空值。"
            if not isinstance(amplitude, (int, float)) or not isinstance(voltage, (int, float)):
                return False, "输入数据必须为数字类型。"
        self.amplitudes.append(amplitude)
        self.voltages.append(voltage)
        return True, "成功添加数据"

    def fit(self, threshold=0.001, json_file_name="calibration_coefficients.json"):
        """
          拟合幅度和电压数据以获得线性关系。
          返回:
               包含状态码和拟合函数的元组。
        """
        if not self.amplitudes or not self.voltages:
            self.logger.error("幅度和电压不能为空")
            return False, "幅度和电压不能为空"
        if len(self.amplitudes) != len(self.voltages):
            self.logger.error("幅度和电压的数量必须一致")
            return False, "幅度和电压的数量必须一致"
        coefficients, residuals, *_ = np.polyfit(self.voltages, self.amplitudes, 1, full=True)
        if len(self.voltages) > 2:
            if len(residuals) == 0:
                self.logger.error("残差为空")
                return False, "残差为空，请重新校准。"
            mse = np.nanmean(residuals ** 2)
            if mse > threshold or mse < 0 or not np.isfinite(mse):
                self.logger.error("校准精度不足，请重新校准。")
                return False, "校准精度不足，请重新校准。"
        save_code, msg = self.save_coefficients_to_json(coefficients, max(self.voltages), json_file_name)
        if save_code == True:
            return True, coefficients
        return save_code, msg

    @staticmethod
    def predict_amplitude(coefficients, target_voltage):
        """
            根据拟合函数和目标电压预测对应的幅度。
            参数:
                coefficients: list
                    拟合系数。
                target_voltage: int 或 float
                    目标电压值。
            返回:
                predict_amplitude: float
                    预测幅度（四位小数）。
        """
        fit_function = np.poly1d(coefficients)
        predict_amplitude = fit_function(target_voltage)
        return np.round(predict_amplitude, 4)

    def calibrate_amplitude(self, target_voltage, json_file_name="calibration_coefficients.json"):
        """
          根据目标电压和保存的校准系数预测幅度。
          参数:
              target_voltage: int 或 float 或 list
              json_file_name: str
                  校准系数的 JSON 文件名。
          返回:
              predict_amplitude: int 或 float 或 list
                  对应目标电压的预测幅度。
        """
        load_code,  load_data = self.load_data_from_json(json_file_name)
        if load_code == True:
            coefficients_data = load_data.get("calibration_coefficients")
            max_voltage = load_data.get("max_voltage")
            predict_amplitude = self.predict_amplitude(coefficients_data, target_voltage)
            return True, (predict_amplitude, max_voltage)
        self.logger.error("加载校准系数失败，请先进行校准")
        return False, "加载校准系数失败，请先进行校准"

    def save_coefficients_to_json(self, coefficients, max_voltages, json_file_name):
        """
            将校准系数和最大电压保存到 JSON 文件。
            参数:
                coefficients: list 或 np.ndarray
                    要保存的校准系数。
                max_voltages: int 或 float
                    要保存的最大电压。
                json_file_name: str
                    保存的 JSON 文件名。
            返回:
                包含状态码和提示信息的元组。
        """
        if not json_file_name.endswith('.json'):
            json_file_name = os.path.splitext(json_file_name)[0]
            json_file_name += '.json'
        json_file_path = utils.get_config_path(json_file_name)
        directory = os.path.dirname(json_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not isinstance(coefficients, (list, np.ndarray)):
            return False, "系数必须是 list 或 numpy array。"
        coefficients = coefficients.tolist() if isinstance(coefficients, np.ndarray) else coefficients
        data = {
            "calibration_coefficients": coefficients,
            "max_voltage": max_voltages
        }
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=3)
                self.logger.info(f"系数已保存至 {json_file_path}.")
                return True, f"成功保存系数到 {json_file_path}."
        except Exception as e:
            err_msg = "保存系数到 JSON 失败。%s" % (str(e)[:50])
            self.logger.error(err_msg)
            return False, err_msg

    def load_data_from_json(self, json_file_name):
        """
            从 JSON 文件中加载校准系数和电压。
            参数:
                json_file_name: str
                    JSON 文件名。
            返回:
                包含状态码和数据或错误信息的元组。
        """
        json_file_path = utils.get_config_path(json_file_name)
        if not os.path.exists(json_file_path):
            return False, "该 JSON 文件不存在"
        try:
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                return True, data
        except Exception as e:
            err_msg = "从 JSON 加载系数数据失败。%s" % (str(e)[:50])
            self.logger.error(err_msg)
            return False, err_msg
