import json
import os

import pyqtgraph
import numpy as np
import sounddevice as sd

from PyQt5.QtCore import QFile, Qt, QTimer
from PyQt5.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QApplication
from PyQt5.uic import loadUi
from pyqtgraph import mkPen

from control.log_manager import LogManager
from control.utils import utils
from control.utils.audio_session_manager import AudioSessionManager
from control.utils.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from control.utils.streaming_audio_processor import DuplexStreamingPlayRec
from custom.customSignals import sign

class MicAdjustInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.signal_info = None
        self.mic1_pa = None
        self.mic2_pa = None
        self.mic_binding = (0, 1)
        self.mic_deviation_db = (1.0, 1.0)
        self.streaming_buffer = []
        self.stream_instance = None
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._handle_queue_and_update_ui)
        self._stream_chunks = []
        self._ui_scale = 1.0

        self.logger = LogManager.set_log_handler("麦克风校准")
        self.init_ui()
        self.init_fun()
        self.init_voltage_value()

    def _handle_queue_and_update_ui(self):
        """
        仿 SpeakerAnomalyDetection 的写法：
        - 定时器里只做三件事：处理队列、实时更新显示、判断结束并触发收尾
        """
        if self.stream_instance is None:
            return

        # 1) 处理队列：把回调线程塞进来的 chunk 取出来
        chunks = self.stream_instance.process_queue()
        sr = self.stream_instance.sample_rate
        if chunks:
            # 2) 实时显示：只画最近 N 秒（避免越画越卡）
            duration = float(self.signal_info.get("signal_time", 5))
            keep_sr = int(duration * sr)

            # 维护滚动窗口（只保留最近 keep 个采样点）
            new_block = np.vstack(chunks)  # (k,4)
            self.streaming_buffer.append(new_block)  # 用 list 装块更快
            buf = np.vstack(self.streaming_buffer)

            # 把 200ms 换算成采样点数；例如 48000Hz 下，0.2s 就是 9600 个点。
            trim_samples = int(round(utils.trim_time * sr))

            # 实时图最多显示 signal_time 秒；因为后面要裁掉 200ms, 所以缓存先多留 200ms ！！！一般用不到！！！
            if buf.shape[0] > keep_sr + trim_samples:
                buf = buf[-(keep_sr + trim_samples):, :]
                self.streaming_buffer = [buf]

            # 已经录够 200ms 后，从这个累计缓存里buf裁掉最前面的 200ms。
            if buf.shape[0] > trim_samples:
                buf = buf[trim_samples:, :]
            # 还没录够 200ms 时先显示空数据，这样启动爆音不会画到界面上。
            else:
                buf = buf[:0, :]

            # 裁完后有数据才更新曲线；没数据就等下一次定时器刷新。
            if buf.shape[0] > 0:
                # 时间轴按裁掉开头后的数据从 0 秒开始显示。
                total_samples = max(0, self.stream_instance.samples_captured - trim_samples)
                start_sample = max(0, total_samples - buf.shape[0])
                time_axis = (start_sample + np.arange(buf.shape[0])) / sr

                # 只做“时域实时更新”
                pa1, pa2 = MicAdjustInterface.process_chunks(buf, self.mic_binding, self.mic_deviation_db)
                self.update_time_plot(time_axis, pa1, pa2)

        #3) 录音结束：录音停止-> stream停timer-> 收集结果(recording/err)-> stop/释放stream-> (FFT/保存/画图)-> 恢复按钮&释放占用锁
        if not self.stream_instance.is_recording:

            self.stream_timer.stop()

            err = getattr(self.stream_instance, "error", None)
            recording = self.stream_instance.get_recorded_data()  # (N,4)

            self.stream_instance.stop()
            self.stream_instance = None

            try:
                if err:
                    raise RuntimeError(f"流式录音出错：{err}")

                self._handle_recording_fft(recording, sr)
            except Exception as e:
                self.logger.exception("录音后处理失败")
                sign.error_message_signal.emit(f"录音失败：{e}", self)

            finally:
                utils.set_adjust_button_enabled(self.start_adjust_button, True)
                AudioSessionManager.release(self)

    def _handle_recording_fft(self, recording, samplerate):
        """
        流式录音结束后的收尾：
        1) 从多通道 recording 中按 binding_index 拆出 4 路麦克
        2) 计算 SPL/scale，得到 Pa 数据
        3) 饱和提示
        4) 保存 MIC1~MIC4 原始数据到 txt
        5) 最终画时域 + FFT
        6) 恢复按钮
        """
        self.logger.info("录音完成(流式),开始处理音频数据")
        self.logger.info(f"recording shape: {getattr(recording, 'shape', None)}")

        recording = utils.trim_start_samples(recording, samplerate)
        self.logger.info(f"已裁掉开头 {utils.trim_time * 1000:.0f} ms, recording shape: {getattr(recording, 'shape', None)}")
        if recording.shape[0] == 0:
            raise ValueError("裁剪后录音数据为空，请检查录音时长和采样率设置。")

        # 1) 处理录音数据（拆通道 + 计算 real_spl/scale）
        (mic1_data, mic2_data, real_spl1, real_spl2, scale1, scale2) = MicAdjustInterface.process_mic_channels_data(
            recording, self.mic_binding, self.mic_deviation_db)

        # 2) 转成 Pa
        self.mic1_pa = mic1_data * scale1
        self.mic2_pa = mic2_data * scale2

        # 3) 饱和提示
        if real_spl1 > 120:
            self.logger.error(f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 120 dB，已饱和")
            QMessageBox.warning(self, "警告", f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 120 dB，已饱和")
        if real_spl2 > 120:
            self.logger.error(f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 120 dB，已饱和")
            QMessageBox.warning(self, "警告", f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 120 dB，已饱和")

        # 4) 保存 txt（保存的是 mic*_data 原始通道数据）
        cal_dir = os.path.join(os.getcwd(), "！校准")
        os.makedirs(cal_dir, exist_ok=True)

        mic1_txt = os.path.join(cal_dir, "MIC1.txt")
        mic2_txt = os.path.join(cal_dir, "MIC2.txt")

        np.savetxt(mic1_txt, mic1_data)
        np.savetxt(mic2_txt, mic2_data)

        self.logger.info(f"MIC1 校准已保存: {mic1_txt}")
        self.logger.info(f"MIC2 校准已保存: {mic2_txt}")

        # 5) 最终画图（时域（pa）+FFT（原始数据））
        duration = float(self.signal_info.get("signal_time", 0))
        if duration <= 0:
            # 如果没有 signal_time，就用数据长度反推
            duration = len(self.mic1_pa) / float(samplerate)

        time_axis = np.linspace(0, duration, len(self.mic1_pa))
        self.update_plot_and_fft(time_axis, mic1_data, mic2_data, samplerate)
        self.logger.info("画图完成")

    def init_ui(self):
        ui_file = QFile(":ui/mic_adjust.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 mic_adjust.ui")
            raise FileNotFoundError("未找到资源文件 mic_adjust.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("麦克风校准")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint| Qt.WindowMaximizeButtonHint)
        # 先按屏幕可用区域缩小窗口
        self._ui_scale = utils.resize_by_ui_with_screen(self)
        self.apply_scaled_layout()
        self.init_images()
        self.init_view()
        self.show()
        self.logger.info("打开麦克风校准界面")

    def _scaled(self, value, minimum=1):
        # 将设计稿里的固定尺寸按窗口缩放比例换算，并保留一个最小可读/可点尺寸。
        return max(minimum, int(round(value * self._ui_scale)))

    def apply_scaled_layout(self):
        # scale 为 1 时说明窗口无需缩小，保持原始 UI 布局。
        if self._ui_scale >= 1:
            return

        # 压缩布局边距和间距，给图表和底部操作区留出更多可见空间。
        root_layout = self.layout()
        if root_layout is not None:
            margin = self._scaled(9, 2)
            root_layout.setContentsMargins(margin, margin, margin, margin)
            root_layout.setSpacing(self._scaled(6, 2))
            for index in range(root_layout.count()):
                root_layout.setStretch(index, 0)

        for layout_name in (
                "horizontalLayout_3", "horizontalLayout_4", "horizontalLayout_5", "horizontalLayout_6",
                "verticalLayout", "verticalLayout_2", "verticalLayout_3", "verticalLayout_4"):
            layout = getattr(self.ui, layout_name, None)
            if layout is not None:
                layout.setSpacing(self._scaled(6, 2))

        # 小分辨率下让说明文字换行，避免把界面撑出窗口。
        for label in (self.label_5, self.label_6):
            label.setWordWrap(True)
            label.setMaximumWidth(self._scaled(145, 72))
            label.setMinimumHeight(self._scaled(120, 90))

        for label in (self.label, self.label_2):
            label.setMaximumWidth(self._scaled(70, 45))

        for frame in (self.frame_plot3, self.frame_plot3_2, self.frame_plot4, self.frame_plot4_2):
            frame.setMinimumSize(self._scaled(170, 120), self._scaled(120, 76))
            frame.setMaximumHeight(self._scaled(128, 84))

    def _apply_plot_screen_size(self, plot):
        # pyqtgraph 控件也要限制高度，否则内部最小高度会把窗口重新撑大。
        if self._ui_scale < 1:
            plot.setMinimumHeight(self._scaled(112, 68))
            plot.setMaximumHeight(self._scaled(120, 76))

    def init_fun(self):
        self.start_adjust_button.clicked.connect(self.start_adjust)
        # 连接失去焦点事件
        self.target_voltage_value.focusOutEvent = self.focusOutEvent  # 将自定义的事件绑定到控件上

        sign.play_adjust_audio_sign.connect(self.play_adjust_audio,Qt.AutoConnection)
        sign.error_message_signal.connect(utils.show_error_message)
        sign.update_plot_sign.connect(self.update_plot)

    def init_voltage_value(self):
        ok, config = utils.get_config_content("output_signal_setting.json")
        if not ok:
            self.logger.error("读取配置失败: output_signal_setting.json")
            QMessageBox.warning(self, "错误", "读取配置失败：output_signal_setting.json")
            return
        mic_adj_v = config.get("mic_adjust_voltage", None)
        self.target_voltage_value.setText(str(mic_adj_v if mic_adj_v is not None else "0.01"))

    def focusOutEvent(self, event):
        """
        当控件失去焦点时触发，进行电压值的校验和保存操作。
        """
        self.validate_and_save_voltage()  # 触发校验和保存操作

        # 调用父类的 focusOutEvent（如果需要其他默认行为）
        super().focusOutEvent(event)

    def validate_and_save_voltage(self):
        text = self.target_voltage_value.text().strip()

        try:
            value = float(text)
            if value < 0:
                raise ValueError("电压必须大于 0")

            ok, config = utils.get_config_content("output_signal_setting.json")
            if ok:
                config["mic_adjust_voltage"] = value
                if utils.write_config_content("output_signal_setting.json", config):
                    self.logger.info(f"保存 mic_adjust_voltage = {value}")
                else:
                    QMessageBox.warning(self, "保存失败", "配置文件保存失败，请重试！")
            else:
                QMessageBox.warning(self, "读取失败", "配置文件读取失败，请检查文件是否存在！")

        except Exception as e:
            # 回退到上次保存的合法值
            QMessageBox.warning(self, "输入错误", f"请输入正确的有效数字: {e}")
            ok, cfg = utils.get_config_content("output_signal_setting.json")
            if ok:
                last_value = cfg.get("mic_adjust_voltage", 0.01)  # 默认值为 0.01
                self.target_voltage_value.setText(str(last_value))
            else:
                self.target_voltage_value.setText("0.01")

    def init_images(self):
        # 缩小时示意图控件也跟随 scale 调整，避免顶部区域把窗口宽度撑回去。
        if self._ui_scale < 1:
            preview_size = (self._scaled(340, 180), self._scaled(255, 130))
        else:
            preview_size = (400, 300)

        utils.set_clickable_graphics_image(
            parent=self,
            view=self.graphicsView,
            image_path=":/images/2mic薄层校准_画板2026_hd.png",
            preview_size=preview_size,
            title="校准图片预览",
        )
        utils.set_clickable_graphics_image(
            parent=self,
            view=self.graphicsView_2,
            image_path=":/images/2mic薄层_画板2026_hd.png",
            preview_size=preview_size,
            title="阻抗管图片预览",
        )
        for view in (self.graphicsView, self.graphicsView_2):
            view.setFixedSize(preview_size[0] + 4, preview_size[1] + 4)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def init_view(self):
        # Mic1 - Time
        self.plot3 = pyqtgraph.PlotWidget()
        self._apply_plot_screen_size(self.plot3)
        layout3 = QVBoxLayout()
        layout3.setContentsMargins(0, 0, 0, 0)
        layout3.addWidget(self.plot3)
        self.ui.frame_plot3.setLayout(layout3)
        self.plot3.setBackground('white')
        self.plot3.setLabel('bottom', 'Time', units='s')
        self.plot3.setLabel('left', 'Amplitude', units='Pa')
        self.plot3.getAxis('bottom').setTextPen("black")
        self.plot3.getAxis('left').setTextPen("black")
        self.plot3.showGrid(x=True, y=True, alpha=0.25)
        self.curve3 = self.plot3.plot([], [], pen=mkPen(color=(76, 120, 168)))

        # Mic1 - Freq
        self.plot3_freq = pyqtgraph.PlotWidget()
        self._apply_plot_screen_size(self.plot3_freq)
        lay3f = QVBoxLayout()
        lay3f.setContentsMargins(0, 0, 0, 0)
        lay3f.addWidget(self.plot3_freq)
        self.ui.frame_plot3_2.setLayout(lay3f)
        self.plot3_freq.setBackground('white')
        self.plot3_freq.setLabel('bottom', 'Frequency(Hz)')
        self.plot3_freq.setLabel('left', 'Magnitude')
        self.plot3_freq.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot3_freq.getAxis('left').enableAutoSIPrefix(False)  # 禁用自动转换单位
        self.plot3_freq.getAxis('bottom').setTextPen("black")
        self.plot3_freq.getAxis('left').setTextPen("black")
        self.plot3_freq.showGrid(x=True, y=True, alpha=0.25)
        self.plot3_freq.setLogMode(x=True, y=False)
        # 调整刻度字体大小
        font = pyqtgraph.QtGui.QFont()
        font.setPointSize(self._scaled(11, 8))
        self.plot3_freq.getAxis('bottom').setTickFont(font)
        # 使用自定义的对数坐标轴刻度标签格式
        self.plot3_freq.getAxis('bottom').logTickStrings = utils.custom_log_tick_strings

        # Mic2 - Time
        self.plot4 = pyqtgraph.PlotWidget()
        self._apply_plot_screen_size(self.plot4)
        layout4 = QVBoxLayout()
        layout4.setContentsMargins(0, 0, 0, 0)
        layout4.addWidget(self.plot4)
        self.ui.frame_plot4.setLayout(layout4)
        self.plot4.setBackground('white')
        self.plot4.setLabel('bottom', 'Time', units='s')
        self.plot4.setLabel('left', 'Amplitude', units='Pa')
        self.plot4.getAxis('bottom').setTextPen("black")
        self.plot4.getAxis('left').setTextPen("black")
        self.plot4.showGrid(x=True, y=True, alpha=0.25)
        self.curve4 = self.plot4.plot([], [], pen=mkPen(color=(84, 162, 75)))

        # Mic2 - Freq
        self.plot4_freq = pyqtgraph.PlotWidget()
        self._apply_plot_screen_size(self.plot4_freq)
        lay4f = QVBoxLayout()
        lay4f.setContentsMargins(0, 0, 0, 0)
        lay4f.addWidget(self.plot4_freq)
        self.ui.frame_plot4_2.setLayout(lay4f)
        self.plot4_freq.setBackground('white')
        self.plot4_freq.setLabel('bottom', 'Frequency(Hz)')
        self.plot4_freq.setLabel('left', 'Magnitude')
        self.plot4_freq.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot4_freq.getAxis('left').enableAutoSIPrefix(False)  # 禁用自动转换单位
        self.plot4_freq.getAxis('bottom').setTextPen("black")
        self.plot4_freq.getAxis('left').setTextPen("black")
        self.plot4_freq.showGrid(x=True, y=True, alpha=0.25)
        self.plot4_freq.setLogMode(x=True, y=False)
        # 调整刻度字体大小
        self.plot4_freq.getAxis('bottom').setTickFont(font)
        # 使用自定义的对数坐标轴刻度标签格式
        self.plot4_freq.getAxis('bottom').logTickStrings = utils.custom_log_tick_strings

    def _load_mic_binding_indices(self):
        # 默认顺序
        idx1, idx2 = 0, 1
        dev1, dev2 = 1.0, 1.0
        try:
            path = utils.get_config_path("mic_calibration.json")
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            idx1 = int(cfg.get("In1", {}).get("binding_index", idx1))
            idx2 = int(cfg.get("In2", {}).get("binding_index", idx2))
            dev1 = float(cfg.get("In1", {}).get("deviation_value", dev1))
            dev2 = float(cfg.get("In2", {}).get("deviation_value", dev2))
        except Exception as e:
            self.logger.warning(f"读取 mic_calibration.json 失败，使用默认绑定 0/1：{e}")

        self.mic_binding = (idx1, idx2)
        self.mic_deviation_db = (dev1, dev2)

    def init_config(self):
        config_path = utils.get_config_path("output_signal_setting.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.signal_info = config.get("signal_info", {})
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取配置失败：{e}")

    def record_and_plot(self):
        if not AudioSessionManager.acquire(self):
            QMessageBox.warning(self, "提示", "音频设备正在被其它界面占用，请先停止/关闭其它录音功能。")
            utils.set_adjust_button_enabled(self.start_adjust_button, True)
            return
        self._load_mic_binding_indices()
        try:
            self.curve3.setData([], [])
            self.curve4.setData([], [])
            self.plot3_freq.clear()
            self.plot4_freq.clear()

            self.streaming_buffer = []
            QApplication.processEvents()
            # 获取设备采样率
            samplerate = utils.get_device_info()
            duration = self.signal_info['signal_time']

            # 根据目标电压，校准并生成激励信号
            target_voltage = float(self.target_voltage_value.text())
            self.signal_info['signal_amplitude'] = target_voltage
            result, data, amplitude, msg = utils.generate_calibrated_signal(self.signal_info, samplerate)

            if result is False:
                raise ValueError(f"{msg}")

            input_channels = sd.query_devices(sd.default.device[0], 'input')['max_input_channels']
            output_channels = sd.query_devices(sd.default.device[1], 'output')['max_output_channels']
            print(f"最大输入通道数: {input_channels},输出通道: {output_channels}")
            self.logger.info(f"最大输入通道数: {input_channels},输出通道: {output_channels}")

            if input_channels < 2:
                raise ValueError(f"检测到的输入通道数为 {input_channels},不足 2 个，无法进行完整校准，请选择别的设备！")

            # 把激励信号data转化成 多维数组
            stimulus_data = np.asarray(data, dtype=np.float32)
            if output_channels > 1:
                stimulus_data = np.column_stack([stimulus_data] * output_channels)
            # 单通道时数据是 (N,)，补静音要拼成 (N, channels)，所以转成 (N, 1)。
            if stimulus_data.ndim == 1:
                stimulus_data = stimulus_data.reshape(-1, 1)

            # 开头补 200ms 静音，录制总时长也增加 200ms；
            # 后处理时裁掉这段开头数据，避免启动瞬间爆音进入校准结果。
            trim_samples = int(round(utils.trim_time * samplerate))
            if trim_samples > 0:
                silence = np.zeros((trim_samples, int(output_channels)), dtype=np.float32)
                stimulus_data = np.vstack([silence, stimulus_data])
            record_duration = float(duration) + utils.trim_time

            self.logger.info("开始播放并录音...")
            # 读设备（你已经保存到 basic_params.json）
            ok, config = utils.get_config_content("basic_params.json")
            input_device = config["basic_params"]["input_selected_device_id"]
            output_device = config["basic_params"]["output_selected_device_id"]
            self.logger.info(f"开始流式播放并录音...dev=({input_device},{output_device})")

            self.stream_instance = DuplexStreamingPlayRec()
            self.stream_instance.start(
                stimulus_data=stimulus_data,
                sample_rate=samplerate,
                input_device=input_device,
                output_device=output_device,
                input_channels=int(input_channels),
                output_channels=int(output_channels),
                duration=record_duration,
                blocksize=2048,
            )
            self.stream_timer.start(50)  # 50ms 刷一次队列/判断结束
            return
        except Exception as e:
            self.logger.exception("录音启动失败")
            sign.error_message_signal.emit(f"录音失败：{e}", self)
            utils.set_adjust_button_enabled(self.start_adjust_button, True)
            AudioSessionManager.release(self)

    @staticmethod
    def process_chunks(buf, mic_binding, mic_deviation_db):
        """
        处理chunks一段数据
        """
        mic1_data = buf[:, mic_binding[0]]
        mic2_data = buf[:, mic_binding[1]]
        spl_smooth1, _ = AudioThdFrequencyResponseAnalysis.spl_calculation(buf[:, mic_binding[0]])
        real_1 = np.max(spl_smooth1)
        spl_smooth2, _ = AudioThdFrequencyResponseAnalysis.spl_calculation(buf[:, mic_binding[1]])
        real_2 = np.max(spl_smooth2)
        factor1, factor2 = mic_deviation_db
        real_spl1 = real_1 + 20 * np.log10(factor1)
        real_spl2 = real_2 + 20 * np.log10(factor2)
        rms1 = utils.calculate_rms(buf[:, mic_binding[0]])
        rms2 = utils.calculate_rms(buf[:, mic_binding[1]])
        scale1 = utils.calculate_scale(real_spl1, rms1)
        scale2 = utils.calculate_scale(real_spl2, rms2)
        pa1 = mic1_data * scale1
        pa2 = mic2_data * scale2
        return pa1, pa2

    @staticmethod
    def process_mic_channels_data(recording, mic_binding, mic_deviation_db):
        """
        处理麦克风通道数据
        """
        mic1_data = recording[:, mic_binding[0]]
        mic2_data = recording[:, mic_binding[1]]
        spl_smooth1, _ = AudioThdFrequencyResponseAnalysis.spl_calculation(recording[:, mic_binding[0]])
        real_1 = np.max(spl_smooth1)
        spl_smooth2, _ = AudioThdFrequencyResponseAnalysis.spl_calculation(recording[:, mic_binding[1]])
        real_2 = np.max(spl_smooth2)
        factor1, factor2 = mic_deviation_db
        real_spl1 = real_1 + 20 * np.log10(factor1)
        real_spl2 = real_2 + 20 * np.log10(factor2)
        print(f"real_1实测: {real_1}, 校准后: {real_spl1}")
        print(f"real_2实测: {real_2}, 校准后: {real_spl2}")
        rms1 = utils.calculate_rms(recording[:, mic_binding[0]])
        rms2 = utils.calculate_rms(recording[:, mic_binding[1]])
        scale1 = utils.calculate_scale(real_spl1, rms1)
        scale2 = utils.calculate_scale(real_spl2, rms2)
        print(f"RMS: {rms1:.8f}, {rms2:.8f}")
        print(f"Scale: {scale1:.2f}, {scale2:.2f}")
        return mic1_data, mic2_data, real_spl1, real_spl2, scale1, scale2

    def start_adjust(self):
        utils.set_adjust_button_enabled(self.start_adjust_button, False)
        self.init_config()
        sign.play_adjust_audio_sign.emit()

    def play_adjust_audio(self):
        self.record_and_plot()

    def update_plot(self, time, mic1_data, mic2_data, samplerate):
        self.plot3.plot(time, mic1_data, pen=mkPen(color=(200, 200, 200)))
        self.plot4.plot(time, mic2_data, pen='gray')
        if samplerate is not None:
            f1, m1 = utils.compute_fft(mic1_data, samplerate)
            f2, m2 = utils.compute_fft(mic2_data, samplerate)

            # 只显示到 Nyquist（rfft 本身就是正频段）
            self.plot3_freq.plot(f1, m1, pen=mkPen(color=(200, 200, 200)))
            self.plot4_freq.plot(f2, m2, pen='gray')

            self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
            self.plot4.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
            self.plot3_freq.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
            self.plot4_freq.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整

    def update_time_plot(self, time, mic1_pa, mic2_pa):
        self.curve3.setData(time, mic1_pa)
        self.curve4.setData(time, mic2_pa)

    def update_plot_and_fft(self, time, mic1_data, mic2_data, samplerate):
        # 时域图
        self.update_time_plot(time, self.mic1_pa, self.mic2_pa)

        # fft
        f1, m1 = utils.compute_fft(mic1_data, samplerate)
        f2, m2 = utils.compute_fft(mic2_data, samplerate)
        # 只显示到 Nyquist（rfft 本身就是正频段）
        # 且限制频率范围为 20–20000 Hz
        f1_range = (f1 >= 20) & (f1 <= 20000)
        f2_range = (f2 >= 20) & (f2 <= 20000)
        self.plot3_freq.plot(f1[f1_range], m1[f1_range], pen=mkPen(color=(76, 120, 168)))
        self.plot4_freq.plot(f2[f2_range], m2[f2_range], pen=mkPen(color=(84, 162, 75)))

        self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
        self.plot4.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
        self.plot3_freq.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
        self.plot4_freq.plotItem.enableAutoRange(axis='xy', enable=True)

    def closeEvent(self, event):
        try:
            if self.stream_timer.isActive():
                self.stream_timer.stop()
            if self.stream_instance is not None:
                self.stream_instance.stop()
                self.stream_instance = None
        except Exception:
            pass
        AudioSessionManager.release(self)  # 关键：释放“声卡占用权”

        if self.parent:
            self.parent.mic_window = None
        sign.update_plot_sign.disconnect(self.update_plot)
        sign.error_message_signal.disconnect(utils.show_error_message)
        sign.play_adjust_audio_sign.disconnect(self.play_adjust_audio)
        self.logger.info("关闭麦克风校准界面")
        event.accept()
