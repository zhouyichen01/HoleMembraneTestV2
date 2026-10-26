import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
import pyqtgraph
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, Qt, QTimer, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox, \
    QVBoxLayout, QMenu, QAction, QFileDialog, QInputDialog, QColorDialog
from PyQt5.uic import loadUi
from pyqtgraph import mkPen
from scipy.signal import savgol_filter

from control.config_tree_interface import ConfigTreeInterface
from control.imp_tube_params_setting_interface import ImptubeParamsSetInterface
from control.log_manager import LogManager
from control.mic_adjust_interface import MicAdjustInterface
from control.output_signal_setting_interface import OutputSignalSetInterface
from control.output_voltage_interface import MicoutputVoltageInterface, SoundcardCalibrationManager
from control.select_deivce_interface import SelectDeviceInterface
from control.utils import utils
from control.utils.audio_session_manager import AudioSessionManager
from control.utils.streaming_audio_processor import DuplexStreamingPlayRec
from custom.customSignals import sign
from resources import icons_rc, ui_rc # 导入资源


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device_window = None
        self.imptube_window = None
        self.output_window = None
        self.mic_window = None
        self.config_tree_window = None
        self.output_voltage_window = None
        self.signal_info = None
        self.tube_params = None
        self.test_result = None
        self.soft_value = None
        self.plot_state = False
        self.mic1_data = None
        self.mic2_data = None
        self.mic1_pa = None
        self.mic2_pa = None
        self.logger = LogManager.set_log_handler("主窗口")
        # ===== 流式录音相关 =====
        self.streaming_buffer = []
        self.stream_instance = None
        self.mic_binding = (0, 1)
        self.mic_deviation_db = (1.0, 1.0)
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._handle_queue_and_update_ui)

        self.init_ui()
        self.init_fun()
        self.init_image()
        self.init_view()
        self.load_basic_params_config()
        self.init_slider()
        self.history_line = []
        # 缓存已读取的历史曲线数据，避免切换图表类型时反复读取 npz 文件。
        self._history_line_data_cache = {}
        self.crosshair_enabled = False
        # 当前十字线选中的数据点下标，左右键移动时会用到
        self.selected_point_index = None

    def init_ui(self):
        ui_file = QFile(":ui/window.ui")
        ui_file.open(QFile.ReadOnly)
        loadUi(ui_file, self)
        self.apply_resolution_font_sizes()
        # 设置窗口标题和大小
        self.setWindowTitle('小孔膜片阻抗管测试系统 V2.0')
        self.showMaximized()

        # 设置窗口图标
        self.setWindowIcon(QIcon(':/images/dongyuan.png'))
        # 获取主界面的布局并设置内容的边距
        layout = self.centralWidget().layout()  # 获取 QMainWindow 的中心控件的布局
        if layout:
            layout.setContentsMargins(1, 1, 1, 1)

    def apply_resolution_font_sizes(self):
        screen = QApplication.primaryScreen()
        metrics = utils.get_screen_metrics(screen)
        font_size = utils.get_font_size_for_resolution(
            metrics["width"], metrics["height"]
        )
        print(
            f"[Font] plot_type_selector={font_size}pt "
            f"(screen={metrics['width']}x{metrics['height']})"
        )

        # 同时设置下拉框当前文本和展开列表的字号。
        self.plot_type_selector.setStyleSheet(f"""
            QComboBox {{
                font: bold {font_size}pt "Microsoft YaHei";
            }}
            QComboBox QAbstractItemView {{
                font: bold {font_size}pt "Microsoft YaHei";
            }}
        """)

    def init_fun(self):
        self.action_2.triggered.disconnect()
        self.action_2.triggered.connect(self.open_select_device_interface)
        self.action.triggered.disconnect()
        self.action.triggered.connect(self.open_imptube_params_setting_interface)
        self.action_3.triggered.disconnect()
        self.action_3.triggered.connect(self.open_output_signal_setting_interface)
        self.action_9.triggered.disconnect()
        self.action_9.triggered.connect(self.open_mic_adjust_interface)
        self.action_6.triggered.disconnect()
        self.action_6.triggered.connect(self.open_output_voltage_interface)
        self.action_8.triggered.disconnect()
        self.action_8.triggered.connect(self.open_config_tree_interface)
        self.action_14.triggered.disconnect()
        self.action_14.triggered.connect(self.popup_pdf)
        self.run_test_button.clicked.connect(self.run_test)
        self.save_test_data.triggered.disconnect()
        self.save_test_data.triggered.connect(self.save_test_data_to_excel)
        self.plot_type_selector.currentIndexChanged.connect(self.update_plot3_by_selector)
        sign.update_plot3_by_selector_sign.connect(self.update_plot3_by_selector)

    def only_view_all_menu(self):
        def context_menu(event):
            menu = QMenu()

            # 添加 View All 操作（等价于原生行为）
            view_all_action = QAction("View All")
            menu.addAction(view_all_action)

            # 设置触发行为：缩放视图范围以适应所有数据
            view_all_action.triggered.connect(lambda: self.plot3.plotItem.enableAutoRange())

            menu.exec_(event.screenPos())

        self.plot3.plotItem.scene().contextMenuEvent = context_menu

    def init_image(self):
        utils.set_clickable_graphics_image(
            parent=self,
            view=self.schematic1,
            image_path=":/images/2mic薄层_画板2026_hd.png",
            preview_size=(435, 291),
            title="首页图片预览",
        )
        # 获取主界面的布局并设置内容的边距
        layout = self.centralWidget().layout()  # 获取 QMainWindow 的中心控件的布局
        if layout:
            layout.setContentsMargins(1, 1, 1, 1)

    def init_view(self):
        self.plot1 = pyqtgraph.PlotWidget()
        layout1 = QVBoxLayout()
        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.addWidget(self.plot1)
        self.frame.setLayout(layout1)
        self.plot1.setBackground('white')
        self.plot1.setLabel('bottom', 'Time', units='s')
        self.plot1.setLabel('left', 'Amplitude', units='Pa')
        self.plot1.getAxis('bottom').setTextPen("black")
        self.plot1.getAxis('left').setTextPen("black")
        self.plot1.showGrid(x=True, y=True, alpha=0.25)
        self.curve1 = self.plot1.plot([], [], pen=mkPen(color=(76, 120, 168)))

        self.plot2 = pyqtgraph.PlotWidget()
        layout2 = QVBoxLayout()
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.addWidget(self.plot2)
        self.frame_2.setLayout(layout2)
        self.plot2.setBackground('white')
        self.plot2.setLabel('bottom', 'Time', units='s')
        self.plot2.setLabel('left', 'Amplitude', units='Pa')
        self.plot2.getAxis('bottom').setTextPen("black")
        self.plot2.getAxis('left').setTextPen("black")
        self.plot2.showGrid(x=True, y=True, alpha=0.25)
        self.curve2 = self.plot2.plot([], [], pen=mkPen(color=(84, 162, 75)))

        self.plot3 = pyqtgraph.PlotWidget()
        layout3 = QVBoxLayout()
        layout3.setContentsMargins(0, 0, 0, 0)
        layout3.addWidget(self.plot3)
        self.frame_3.setLayout(layout3)
        self.plot3.setBackground('white')
        self.plot3.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plot3.customContextMenuRequested.connect(self.on_plot3_menu)

        self.plot3.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot3.getAxis('left').enableAutoSIPrefix(False) # 禁用自动转换单位
        self.plot3.getAxis('bottom').setLogMode(True)  # 对数坐标轴，适用于频率
        self.plot3.getAxis('left').setLogMode(False)  # 纵轴线性，负值可直接显示
        # 使用自定义的对数坐标轴刻度标签格式
        self.plot3.getAxis('bottom').logTickStrings = utils.custom_log_tick_strings
        # self.plot3.getAxis('bottom').setTickSpacing(levels=[(1, 1)])
        # 设置网格线和背景
        self.plot3.getPlotItem().showGrid(x=True, y=True)

        # 调整刻度字体大小
        font = pyqtgraph.QtGui.QFont()
        font.setPointSize(12)
        self.plot3.getAxis('bottom').setTickFont(font)
        self.plot3.getAxis('left').setTickFont(font)
        self.plot3.setLabel('left', 'Z abs', units='Rayl')
        self.plot3.setLabel('bottom', 'Freq', units='Hz')
        self.plot3.getAxis('bottom').setTextPen("black")
        self.plot3.getAxis('left').setTextPen("black")
        # 清空范围限制，允许滚轮缩放
        self.plot3.setAutoVisible(True)  # 启用自动范围调整
        # 让 plot3 能接收键盘事件，这样双击选点后可以按左右键移动
        self.plot3.setFocusPolicy(Qt.StrongFocus)
        self.plot3.installEventFilter(self)
        self.plot3.plotItem.scene().sigMouseMoved.connect(self.mov)
        # legend 只创建一次
        self.legend = self.plot3.addLegend(offset=(-10, 10))
        # 主曲线（只建一次）
        self.curve3 = self.plot3.plot([], [], pen=mkPen(color='black', width=2), name='本次测试')

    def on_plot3_clicked(self, event):
        if event.double():
            mouse_point = self.plot3.getViewBox().mapSceneToView(event.scenePos())
            x_log_clicked = mouse_point.x()
            freq_clicked = 10 ** x_log_clicked  # 实际频率

            data = self.get_current_plot_data()
            if data is None:
                return
            freq_array, _, _ = data

            # 找出频率中最接近点击频率的点
            index = np.abs(freq_array - freq_clicked).argmin()
            self.logger.info(f"双击事件: 频率(x): {freq_clicked:.2f} Hz, 最近数据索引: {index}")
            self.show_crosshair_at_index(index)
            # 焦点给图表，否则左右键可能会被其它控件吃掉
            self.plot3.setFocus()

    def get_current_plot_data(self):
        """
        获取当前下拉框对应曲线的数据。
        返回原始频率、log10(x)、y，十字线显示和左右键移动都复用它。
        """
        if not self.test_result:
            return None

        plot_type = self.plot_type_selector.currentText()

        if plot_type == "传输阻抗率Z abs":
            # 获取当前显示的曲线数据
            freq_array = np.asarray(self.test_result["f"], dtype=float)
            y_array = self.test_result["Z_abs"]
        elif plot_type == "传输阻抗率Z Re":
            freq_array = np.asarray(self.test_result["f"], dtype=float)
            y_array = self.test_result["Z_Re"]
        elif plot_type == "传输阻抗率Z Im":
            freq_array = np.asarray(self.test_result["f"], dtype=float)
            y_array = self.test_result["Z_Im"]
        elif plot_type == "质点速度V abs":
            freq_array = np.asarray(self.test_result["f_v"], dtype=float)
            y_array = self.test_result["V_abs"]
        elif plot_type == "质点速度V Re":
            freq_array = np.asarray(self.test_result["f_v"], dtype=float)
            y_array = self.test_result["V_Re"]
        elif plot_type == "质点速度V Im":
            freq_array = np.asarray(self.test_result["f_v"], dtype=float)
            y_array = self.test_result["V_Im"]
        else:
            return None

        # 如果 freq_array 里面一个频率点都没有，就直接返回 None
        if len(freq_array) == 0:
            return None

        # 这里返回线性 y_disp。
        y_disp = np.asarray(y_array, dtype=float).copy()
        x_disp = np.log10(freq_array)
        # 如果启用了平滑处理
        if self.soft_value > 3:
            y_disp = savgol_filter(y_disp, window_length=self.soft_value, polyorder=3)

        return freq_array, x_disp, y_disp

    def show_crosshair_at_index(self, index):
        """
        把十字线、小黄点和提示框移动到指定的数据点。
        index 是数据点下标，不是频率值。
        """
        data = self.get_current_plot_data()
        if data is None:
            return

        freq_array, x_disp, y_disp = data
        # 防止按到最左/最右后越界
        index = max(0, min(int(index), len(freq_array) - 1))
        self.selected_point_index = index
        x_log = float(x_disp[index])  # 显示坐标（log10 频率）
        # y_log = float(y_disp[index])
        x_lin = float(freq_array[index])  # 线性频率x
        # 幅值y, 取平滑后的显示数据, 而非原始数据
        y_lin = float(y_disp[index])

        self.logger.info(f"对数x频率: {x_log:.4f}), 对数无y幅值")
        self.logger.info(f"线性x频率: {x_lin:.2f} Hz, 线性y幅值: {y_lin:.4f}")
        # 初始化十字线和文字提示
        if not self.crosshair_enabled:
            self.vLine = pyqtgraph.InfiniteLine(angle=90, movable=False, pen='r')
            self.hLine = pyqtgraph.InfiniteLine(angle=0, movable=False, pen='r')
            self.text = pyqtgraph.TextItem("", anchor=(0, 1), fill=pyqtgraph.mkBrush(255, 255, 255, 200),
                                           border='k')
            self.plot3.addItem(self.vLine, ignoreBounds=True)
            self.plot3.addItem(self.hLine, ignoreBounds=True)
            self.plot3.addItem(self.text)
            self.crosshair_enabled = True

        self.vLine.setPos(x_log)
        self.hLine.setPos(y_lin)

        # 添加点击标记
        if hasattr(self, "click_marker") and self.click_marker is not None:
            self.plot3.removeItem(self.click_marker)
        self.click_marker = pyqtgraph.ScatterPlotItem(
            [x_log], [y_lin],
            symbol='o', size=4, brush=pyqtgraph.mkBrush(255, 255, 0), pen='k'
        )
        self.plot3.addItem(self.click_marker)

        # 提示框（显示线性频率和与图一致的线性幅值）
        self.text.setHtml(
            f"<div style='background-color:white; padding:2px;'>"
            f"<b>频率(x):</b> {x_lin:.2f} Hz<br>"
            f"<b>幅值(y):</b> {y_lin:.4f}</div>"
        )
        self.text.setPos(x_log, y_lin)

    def move_selected_point(self, step):
        """
        左右键移动当前选中点。
        step=-1 表示左移一个数据点，step=1 表示右移一个数据点。
        """
        if not self.crosshair_enabled or self.selected_point_index is None:
            return False

        data = self.get_current_plot_data()
        if data is None:
            return False

        freq_array, _, _ = data
        new_index = self.selected_point_index + step
        new_index = max(0, min(new_index, len(freq_array) - 1))
        self.show_crosshair_at_index(new_index)
        return True

    def eventFilter(self, obj, event):
        # 图表获得焦点时，优先在这里处理左右键
        if obj is self.plot3 and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Left and self.move_selected_point(-1):
                return True
            if event.key() == Qt.Key_Right and self.move_selected_point(1):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        # 兜底处理：如果焦点在主窗口，也能用左右键移动十字线
        if event.key() == Qt.Key_Left and self.move_selected_point(-1):
            return
        if event.key() == Qt.Key_Right and self.move_selected_point(1):
            return
        super().keyPressEvent(event)

    def on_plot3_menu(self, pos):
        menu = self.plot3.getPlotItem().getViewBox().menu

        # 去掉“Export/导出”相关子菜单/动作，仅保留 X axis / Y axis 等
        for act in list(menu.actions()):
            try:
                text = act.text()
            except Exception:
                text = ""
            if text and ("Export" in text or "Options" in text or "Mouse" in text):
                menu.removeAction(act)

        # 确保“保存当前曲线”只添加一次；若已存在则仅保证信号连接正确
        existing_save_action = None
        existing_range_action = None
        existing_reset_range_action = None
        for i in menu.actions():
            if i.text() == "保存当前曲线":
                existing_save_action = i
            elif i.text() == "设置显示范围":
                existing_range_action = i
            elif i.text() == "恢复全部范围":
                existing_reset_range_action = i

        if existing_save_action is None:
            act = QAction("保存当前曲线", self)
            act.triggered.connect(self.save_test_result_to_json)
            menu.addAction(act)
        if existing_range_action is None:
            act = QAction("设置显示范围", self)
            act.triggered.connect(self.set_plot3_display_range)
            menu.addAction(act)

        if existing_reset_range_action is None:
            act = QAction("恢复全部范围", self)
            act.triggered.connect(lambda: self.plot3.plotItem.enableAutoRange(axis='xy', enable=True))
            menu.addAction(act)
        # 弹出完整菜单（自带 X axis 等），自定义按钮也在其中
        menu.exec_(self.plot3.mapToGlobal(pos))

    def set_plot3_display_range(self):
        text, ok = QInputDialog.getText(
            self,
            "设置显示范围",
            "输入范围：X最小,X最大；可选 Y最小,Y最大\n例如：100,200 或 100,200;1000,20000"
        )
        if not ok:
            return

        text = text.strip().replace("；", ";").replace("，", ",")
        if not text:
            return

        parts = [i.strip() for i in text.split(";")]
        try:
            if parts[0]:
                x_min, x_max = self._parse_positive_range(parts[0], "X")
                self.plot3.setXRange(np.log10(x_min), np.log10(x_max), padding=0)

            if len(parts) > 1 and parts[1]:
                y_min, y_max = self._parse_positive_range(parts[1], "Y")
                self.plot3.setYRange(np.log10(y_min), np.log10(y_max), padding=0)

        except ValueError as e:
            QMessageBox.warning(self, "提示", str(e))

    @staticmethod
    def _parse_positive_range(text, axis_name):
        values = [i.strip() for i in text.split(",")]
        if len(values) != 2:
            raise ValueError(f"{axis_name}范围格式应为：最小值,最大值")

        start = float(values[0])
        end = float(values[1])
        if start <= 0 or end <= 0:
            raise ValueError(f"{axis_name}范围必须大于 0")
        if start >= end:
            raise ValueError(f"{axis_name}最小值必须小于最大值")
        return start, end

    def save_test_result_to_json(self):
        """
        右键“保存当前曲线”：
        - 询问名称、颜色（用于左侧历史数据设置展示）
        - 大数组保存到独立 npz 文件，tree_config.json 只保存管理信息和文件引用
        """
        if not self.test_result:
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return

        # 1) 输入名称（默认给一个）
        default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name, ok = QInputDialog.getText(self, "保存历史线名称", "请输入名称：", text=default_name)
        if not ok:
            return
        if not name.strip():
            QMessageBox.warning(self, "提示", "名称不能为空！")
            return
        name = name.strip()

        # 2) 选择颜色
        color = QColorDialog.getColor(parent=self, title="选择颜色")
        if not color.isValid():
            return
        colour_str = color.name()  # 例如 "#1e90ff"

        # 3) 读取配置
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            self.logger.error("读取 tree_config.json 失败")
            QMessageBox.warning(self, "错误", "读取配置失败，无法写入。")
            return
        # 4) 计算新索引；配置文件初始化时已保证 manage_history_line 存在。
        new_index = len(content["line"]["manage_history_line"])
        if new_index >= 10:
            QMessageBox.warning(self, "提示", f"历史数据已达 10 条，最多 10 条，请先删除后再保存。")
            return

        # 5) 曲线数据数组很大，保存到独立压缩文件；tree_config.json 只记录文件引用。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 使用相对 tree_config.json 的路径，源码运行和打包运行时都能通过配置目录定位。
        data_file = f"history_lines/history_line_{timestamp}_{new_index}.npz"
        if not utils.save_history_line_data("tree_config.json", data_file, self.test_result):
            QMessageBox.critical(self, "错误", "保存历史线数据文件失败，请重试。")
            return

        # 刚保存的数据也放进内存缓存，保存后刷新图表时不用立刻再读一次磁盘。
        self._history_line_data_cache[data_file] = {
            key: np.asarray(value) for key, value in self.test_result.items()
        }

        # 6) 展示用信息
        content["line"]["manage_history_line"][str(new_index)] = {
            "state": "True",  # 默认显示
            "colour": colour_str,  # 用户选择的颜色
            "line_name": name,  # 用户输入的名字
            "data_file": data_file  # 当前历史线对应的 npz 数据文件
        }

        # 7) 写回
        ok = utils.write_config_content("tree_config.json", content)
        if ok:
            QMessageBox.information(self, "成功", f"已保存历史线：{name}\n颜色：{colour_str}\n索引：{new_index}")
        else:
            # 如果配置写入失败，删除刚生成但没有配置引用的数据文件。
            utils.delete_history_line_data("tree_config.json", data_file)
            QMessageBox.critical(self, "错误", "写入 tree_config.json 失败，请重试。")

        # 8) 刷新配置界面
        if self.config_tree_window and self.config_tree_window.isVisible():
            self.config_tree_window.fill_page_2()
        # 9) 刷新self.plot3
        self.update_plot3_by_selector()
        QApplication.processEvents()

    def mov(self, pos):
        if self.plot3.sceneBoundingRect().contains(pos):
            mouse_point = self.plot3.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            real_x = 10 ** x
            real_y = y  # 纵轴线性，无需换算
            threshold = 1e10
            if real_x > threshold:
                x_str = "∞"
            else:
                x_str = f"{real_x:.2f}"
            if abs(real_y) > threshold:
                y_str = "∞"
            else:
                y_str = f"{real_y:.2f}"
                self.common_pos_value.setText(f"({x_str}, {y_str})")

    def init_slider(self):
        self.slider = self.findChild(QtWidgets.QSlider, "slider")
        # 添加浮动标签
        self.float_label = QLabel(self)
        self.float_label.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.float_label.setFixedSize(60, 20)
        self.float_label.setAlignment(Qt.AlignCenter)
        self.float_label.show()
        self.float_label.raise_()

        # 初始化浮动标签
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setTickInterval(10)
        self.slider.setValue(self.soft_value)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        self.slider.valueChanged.connect(self.update_float_label)

    def _reposition_slider_label(self):
        if hasattr(self, "slider") and self.slider is not None:
            self.update_float_label(self.slider.value())

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._reposition_slider_label)  # 首次显示后再定位

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reposition_slider_label)  # 窗口尺寸变化后再定位

    def update_float_label(self, value):
        float_value = value / 1
        self.float_label.setText(f"{float_value:.1f}")
        self.float_label.setStyleSheet("""
            background-color: #f5f5f5;
            border: 1px solid #999;
            border-radius: 4px;
            padding: 2px;
            font-size: 10pt;
        """)
        # 获取 slider 左上角全局位置 → 相对于 self 的偏移
        slider_pos = self.slider.mapTo(self, self.slider.rect().topLeft())
        slider_width = self.slider.width()
        value_ratio = (value - self.slider.minimum()) / (self.slider.maximum() - self.slider.minimum())

        # 更精确地估计 handle 的像素位置（减去10调整）
        handle_x = int(value_ratio * (slider_width - 20))

        # 设置浮动标签的位置（居中显示）
        self.float_label.move(
            slider_pos.x() + handle_x - self.float_label.width() // 2 + 10,
            slider_pos.y() - self.float_label.height() - 5
        )
        soft_value = int(float(self.float_label.text()))
        self.save_basic_params_config(soft_value)
        self.load_basic_params_config()
        if self.plot_state is True:
            self.update_plot3_by_selector()
        else:
            # 不更新界面
            pass

    def load_basic_params_config(self):
        basic_params_path = utils.get_config_path("basic_params.json")
        try:
            with open(basic_params_path , "r", encoding="utf-8") as f:
                basic_config  = json.load(f)
                self.soft_value = basic_config.get("basic_params").get("soft_value")
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取{basic_params_path}配置失败：{e}")

    def save_basic_params_config(self, soft_value=None, **kwargs):
        basic_params_path = utils.get_config_path("basic_params.json")
        try:
            with open(basic_params_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "basic_params" not in config:
                config["basic_params"] = {}

            config["basic_params"]["soft_value"] = soft_value

            with open(basic_params_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"写入配置失败：{e}")


    def move_to_center(self):
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        # self.resize(w, h)
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

    def open_select_device_interface(self):
        if self.device_window is None:
            self.device_window = SelectDeviceInterface(self)
        self.device_window.show()
        self.device_window.raise_()  # 窗口置顶
        self.device_window.activateWindow()


    def open_imptube_params_setting_interface(self):
        if self.imptube_window is None:
            self.imptube_window = ImptubeParamsSetInterface(self)
        self.imptube_window.show()
        self.imptube_window.raise_()  # 窗口置顶
        self.imptube_window.activateWindow()

    def open_output_signal_setting_interface(self):
        if self.output_window is None:
            self.output_window = OutputSignalSetInterface(self)
        self.output_window.show()
        self.output_window.raise_()
        self.output_window.activateWindow()

    def open_mic_adjust_interface(self):
        if self.mic_window is None:
            self.mic_window = MicAdjustInterface(self)
        self.mic_window.show()
        self.mic_window.raise_()
        self.mic_window.activateWindow()

    def open_output_voltage_interface(self):
        if self.output_voltage_window is None:
            self.output_voltage_window = MicoutputVoltageInterface(self)
        self.output_voltage_window.show()
        self.output_voltage_window.raise_()
        self.output_voltage_window.activateWindow()

    def open_config_tree_interface(self):
        if self.config_tree_window is None:
            self.config_tree_window = ConfigTreeInterface(self)
        self.config_tree_window.show()
        self.config_tree_window.raise_()
        self.config_tree_window.activateWindow()

    def init_config(self):
        output_path = utils.get_config_path("output_signal_setting.json")
        try:
            with open(output_path , "r", encoding="utf-8") as f:
                output_config  = json.load(f)
                self.signal_info = output_config .get("signal_info", {})
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取{output_path}配置失败：{e}")

        tube_path = utils.get_config_path("imp_tube_params_setting.json")
        try:
            with open(tube_path , "r", encoding="utf-8") as f:
                tube_config  = json.load(f)
                self.tube_params = tube_config.get("tube_params", {})

        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取{tube_path}配置失败：{e}")

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

    def run_test(self):
        self.init_config()
        self._load_mic_binding_indices()
        # 防止多个界面同时占用声卡
        if not AudioSessionManager.acquire(self):
            QMessageBox.warning(self, "提示", "音频设备正在被其它界面占用，请先停止/关闭其它录音功能。")
            self.test_state.setText("音频被占用")
            return
        utils.set_run_button_enabled(self.run_test_button, False)
        self.test_state.setText("测试中...")
        QApplication.processEvents()
        self.record_and_plot()

    def record_and_plot(self):
        try:
            # 清空显示（不要 plot().clear() 叠加）
            self.curve1.setData([], [])
            self.curve2.setData([], [])
            self.curve3.setData([], [])
            # 清历史线(legend)也会一起清除
            for line in getattr(self, "history_line", []):
                self.plot3.removeItem(line)
            self.history_line = []
            # 清除 十字线
            if self.crosshair_enabled:
                self.del_cross()
            self.streaming_buffer = []
            QApplication.processEvents()
            # 使用“阻抗管参数设置”中保存的采样率
            samplerate = utils.get_impedance_tube_sample_rate()
            if samplerate not in (44100, 48000):
                raise ValueError("请在“阻抗管参数设置”中选择 44100 Hz 或 48000 Hz 后再测试。")
            duration = self.signal_info['signal_time']
            # 根据目标电压，校准并生成激励信号
            result, data, amplitude, msg = utils.generate_calibrated_signal(self.signal_info, samplerate)
            if result is False:
                self.test_state.setText("缺少输出校准文件")
                QMessageBox.warning(self, "提示", "缺少输出校准文件，请先进行输入/输出校准。")
                utils.set_run_button_enabled(self.run_test_button, True)
                AudioSessionManager.release(self)
                return

            input_channels = sd.query_devices(sd.default.device[0], 'input')['max_input_channels']
            output_channels = sd.query_devices(sd.default.device[1], 'output')['max_output_channels']
            print(f"最大输入通道数: {input_channels},输出通道: {output_channels}")
            self.logger.info(f"最大输入通道数: {input_channels},输出通道: {output_channels}")

            if input_channels < 2:
                raise ValueError(f"检测到的输入通道数为 {input_channels},不足 2 个，无法进行完整校准，请选择别的设备！")

            # 开始录音 (多声道)
            stimulus_data = np.asarray(data, dtype=np.float32)
            if output_channels > 1:
                stimulus_data = np.column_stack([stimulus_data] * output_channels)
            # 单通道时数据是 (N,)，补静音要拼成 (N, channels)，所以转成 (N, 1)。
            if stimulus_data.ndim == 1:
                stimulus_data = stimulus_data.reshape(-1, 1)

            # 先播放 200ms 静音并多录 200ms；后面统一裁掉开头，避开声卡/设备启动爆音。
            trim_samples = int(round(utils.trim_time * samplerate))
            if trim_samples > 0:
                silence = np.zeros((trim_samples, int(output_channels)), dtype=np.float32)
                stimulus_data = np.vstack([silence, stimulus_data])
            record_duration = float(duration) + utils.trim_time

            # 读设备（你已经保存到 basic_params.json）
            ok, config = utils.get_config_content("basic_params.json")
            input_device = config["basic_params"]["input_selected_device_id"]
            output_device = config["basic_params"]["output_selected_device_id"]
            self.logger.info(f"开始流式播放并录音..")

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
            self.test_state.setText(f"测试失败: {e}")
            QMessageBox.critical(self, "错误", f"录音失败：{e}")
            utils.set_run_button_enabled(self.run_test_button, True)
            AudioSessionManager.release(self)

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

            # 已经录够 200ms 后，才把前 200ms 裁掉给界面显示。
            if buf.shape[0] > trim_samples:
                buf = buf[trim_samples:, :]
            else:
                # 还没录够 200ms 时先显示空数据，这样启动爆音不会画到界面上。
                buf = buf[:0, :]

            # 裁完后有数据才更新曲线；没数据就等下一次定时器刷新。
            if buf.shape[0] > 0:
                # 时间轴按裁剪后的数据重新从 0 秒开始。
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
                utils.set_run_button_enabled(self.run_test_button, True)
                AudioSessionManager.release(self)

    def _handle_recording_fft(self, recording, samplerate):
        self.logger.info("录音完成(流式),开始处理音频数据")
        self.logger.info(f"recording shape: {getattr(recording, 'shape', None)}")
        duration = float(self.signal_info.get("signal_time", 0))

        recording = utils.trim_start_samples(recording, samplerate)
        self.logger.info(f"已裁掉开头 {utils.trim_time * 1000:.0f} ms, recording shape: {getattr(recording, 'shape', None)}")
        if recording.shape[0] == 0:
            raise ValueError("裁剪后录音数据为空，请检查录音时长和采样率设置。")

        # 1) 计算（SPL/scale）
        self.mic1_data, self.mic2_data, real_spl1, real_spl2, scale1, scale2 = (
            MicAdjustInterface.process_mic_channels_data(recording, self.mic_binding, self.mic_deviation_db))

        # 2) 转成 Pa
        self.mic1_pa = self.mic1_data * scale1
        self.mic2_pa = self.mic2_data * scale2

        # 3) 饱和提示
        if real_spl1 > 120:
            self.logger.error(f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 {120} dB，已饱和")
            QMessageBox.warning(self, "警告", f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 {120} dB，已饱和")
        if real_spl2 > 120:
            self.logger.error(f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 {120} dB，已饱和")
            QMessageBox.warning(self, "警告", f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 {120} dB，已饱和")


        # 4) 保存 txt（保存的是 mic*_data 原始通道数据）
        save_dir = os.path.join(os.getcwd(), "！采集")
        os.makedirs(save_dir, exist_ok=True)
        mic1_txt = os.path.join(save_dir, "MIC1.txt")
        mic2_txt = os.path.join(save_dir, "MIC2.txt")
        np.savetxt(mic1_txt, self.mic1_data)
        np.savetxt(mic2_txt, self.mic2_data)
        self.logger.info(f"MIC1 采集已保存: {mic1_txt}")
        self.logger.info(f"MIC2 采集已保存: {mic2_txt}")

        # 画最终 Pa 曲线
        time_axis = np.linspace(0, duration, len(self.mic1_pa))
        self.update_time_plot(time_axis, self.mic1_pa, self.mic2_pa)

        try:
            mic1_cal_data, mic2_cal_data = self.load_calibration_data(samplerate)
        except Exception as e:
            self.logger.warning(f"麦克风校准数据不可用: {e}")
            self.test_state.setText("校准数据不可用")
            QMessageBox.warning(self, "提示", f"校准数据不可用：{e}")
            return

        impedance_params = self._get_impedance_tube_parameter_inputs()
        if impedance_params is None:
            self.test_state.setText("参数错误")
            return
        (ff_subset, f_subset, Z_abs, Z_Re, Z_Im,
         V_abs, V_Re, V_Im) = utils.calculate_impedance_Lumped_parameter(
            mic2=self.mic2_data,
            mic1=self.mic1_data,
            mic2_cal=mic2_cal_data,
            mic1_cal=mic1_cal_data,
            rear_mic_voltage=self.mic2_data,
            sf=samplerate,
            **impedance_params,
        )
        self.test_result = {
            "f": ff_subset,
            "f_v": f_subset,
            "Z_abs": Z_abs,
            "Z_Re": Z_Re,
            "Z_Im": Z_Im,
            "V_abs": V_abs,
            "V_Re": V_Re,
            "V_Im": V_Im,
        }
        self.update_plot3_by_selector()

        self.logger.info(f"画图完成！")
        self.test_state.setText("测试完成")
        utils.set_run_button_enabled(self.run_test_button, True)
        AudioSessionManager.release(self)

# ===============
    @staticmethod
    def generate_calibrated_signal(signal_info, samplerate):
        """
        生成校准后的刺激信号
        参数:
            signal_info: dict, 包含信号参数，其中 "signal_amplitude" 表示目标电压幅值 (V)
            samplerate: int, 采样率

        返回:
            success (bool): 是否成功
            data (ndarray): 已按校准幅值缩放后的信号数据
            cal_amplitude (float): 校准后的幅值（实际用于缩放的因子，单位 V）
            error_msg (str): 失败时的错误信息，否则为 None
        """
        params_voltage = signal_info["signal_amplitude"]
        scm = SoundcardCalibrationManager()
        # 用y=ax+b校准函数求幅值因子cal_amplitude
        calibrate_code, calibrate_result = scm.calibrate_amplitude(params_voltage)
        if calibrate_code != True:
            return False, None, None, f"校准幅值失败：{calibrate_result}"
        cal_amplitude, max_voltage = calibrate_result
        if params_voltage > max_voltage:
            return False, None, None, f"参数电压过大（{params_voltage}V），最大可达 {max_voltage}V。"
        # 信号归一化
        signal_info['signal_amplitude'] = 1
        _, data = utils.generate_chirp_wrapper(signal_info, samplerate)
        # 用幅值因子进行缩放
        data = data * cal_amplitude
        return True, data, cal_amplitude, None

    def update_time_plot(self, time, mic1_pa, mic2_pa):
        self.curve1.setData(time, mic1_pa)
        self.curve2.setData(time, mic2_pa)

    @staticmethod
    def load_calibration_data(expected_sample_rate=None):
        # 读取校准数据
        cal_dir = os.path.join(os.getcwd(), "！校准")
        mic1_path = os.path.join(cal_dir, "MIC1.txt")
        mic2_path = os.path.join(cal_dir, "MIC2.txt")
        metadata_path = os.path.join(cal_dir, "recording_metadata.json")

        if expected_sample_rate is not None and os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                calibration_rate = int(json.load(f).get("sample_rate"))
            if calibration_rate != int(expected_sample_rate):
                raise ValueError(
                    f"校准数据采样率为 {calibration_rate} Hz，与当前 {expected_sample_rate} Hz 不一致；"
                    "请在当前采样率下重新进行麦克风校准。"
                )

        # 加载校准数据（float32 精度）
        mic1_cal = np.loadtxt(mic1_path, dtype=np.float32)
        mic2_cal = np.loadtxt(mic2_path, dtype=np.float32)
        return mic1_cal, mic2_cal

    @staticmethod
    def load_record_data():
        # 读取采集数据
        record_dir = os.path.join(os.getcwd(), "！采集")
        mic1_path = os.path.join(record_dir, "MIC1.txt")
        mic2_path = os.path.join(record_dir, "MIC2.txt")

        # 加载校准数据（float32 精度）
        mic1_record = np.loadtxt(mic1_path, dtype=np.float32)
        mic2_record = np.loadtxt(mic2_path, dtype=np.float32)
        return mic1_record, mic2_record

    def save_test_data_to_excel(self):
        if not self.test_result:
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return
        # 获取原始数据
        line_f = self.test_result["f"]

        # 与绘图逻辑一致：全部在线性域平滑，保留符号
        Z_abs = np.asarray(self.test_result["Z_abs"], dtype=float).copy()
        Z_Re = np.asarray(self.test_result["Z_Re"], dtype=float).copy()
        Z_Im = np.asarray(self.test_result["Z_Im"], dtype=float).copy()
        V_abs = np.asarray(self.test_result["V_abs"], dtype=float).copy()
        V_Re = np.asarray(self.test_result["V_Re"], dtype=float).copy()
        V_Im = np.asarray(self.test_result["V_Im"], dtype=float).copy()

        if self.soft_value > 3:
            Z_abs = savgol_filter(Z_abs, window_length=self.soft_value, polyorder=3)
            Z_Re = savgol_filter(Z_Re, window_length=self.soft_value, polyorder=3)
            Z_Im = savgol_filter(Z_Im, window_length=self.soft_value, polyorder=3)
            V_abs = savgol_filter(V_abs, window_length=self.soft_value, polyorder=3)
            V_Re = savgol_filter(V_Re, window_length=self.soft_value, polyorder=3)
            V_Im = savgol_filter(V_Im, window_length=self.soft_value, polyorder=3)

        file1, _ = QFileDialog.getSaveFileName(
            self,
            "保存测试数据",
            os.path.join(os.getcwd(), "传输阻抗率测试结果.csv"),
            "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not file1:
            return  # 用户取消
        if not os.path.splitext(file1)[1]:
            file1 += ".csv"

        try:
            with open(file1, "w", newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["f_Z", "Z_abs", "Z_Re", "Z_Im", "f_V", "V_abs", "V_Re", "V_Im"])
                for i in range(len(line_f)):
                    writer.writerow([
                        line_f[i], Z_abs[i], Z_Re[i], Z_Im[i],
                        self.test_result["f_v"][i], V_abs[i], V_Re[i], V_Im[i],
                    ])

            msg = f"保存成功！\n文件已保存至：\n{file1}"
            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def update_plot3_by_selector(self, source=None):
        text = self.plot_type_selector.currentText()
        is_velocity = text.startswith("质点速度V")
        if "abs" in text:
            label = "V abs" if is_velocity else "Z abs"
            units = "m/s" if is_velocity else "Rayl"
            self.plot3.setLabel('left', label, units=units)
        elif "Re" in text:
            label = "V Re" if is_velocity else "Z Re"
            units = "m/s" if is_velocity else "Rayl"
            self.plot3.setLabel('left', label, units=units)
        elif "Im" in text:
            label = "V Im" if is_velocity else "Z Im"
            units = "m/s" if is_velocity else "Rayl"
            self.plot3.setLabel('left', label, units=units)
        else:
            QMessageBox.warning(self, "提示", f"无{text}选项")
            return

        if not self.test_result:
            if source == "config_tree":
                return
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return

        # 清历史线
        for line in getattr(self, "history_line", []):
            self.plot3.removeItem(line)
        self.history_line = []

        # 清十字线（不然会残留）
        if self.crosshair_enabled:
            self.del_cross()
        # 重新画图后，旧的选中下标可能不属于当前曲线，先清空
        self.selected_point_index = None

        try:
            self.plot3.scene().sigMouseClicked.disconnect(self.on_plot3_clicked)
        except Exception:
            pass
        self.plot3.scene().sigMouseClicked.connect(self.on_plot3_clicked)

        text = self.plot_type_selector.currentText()
        log_f = np.log10(self.test_result["f"])

        if "abs" in text:
            y = np.asarray(self.test_result["Z_abs"], dtype=float)
        elif "Re" in text:
            y = np.asarray(self.test_result["Z_Re"], dtype=float)
        elif "Im" in text:
            y = np.asarray(self.test_result["Z_Im"], dtype=float)
        else:
            QMessageBox.warning(self, "提示", f"无{text}选项")
            return

        if self.soft_value > 3:
            y = savgol_filter(y, window_length=self.soft_value, polyorder=3)
        # 关键：只 setData，不再 plot()
        self.curve3.setData(log_f, y)
        self.curve3.show()
        self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
        # self.plot3.plotItem.enableAutoRange(axis='y', enable=False)  # Y轴范围不变
        self.plot_state = True
        # 画历史线
        self.update_history_lines()

    # def _legacy_update_plot3_by_selector(self):
    #     """
    #     根据平滑值来重新画图
    #     """
    #     text = self.plot_type_selector.currentText()
    #     if text == "传输阻抗率Z abs":
    #         self.plot3.setLabel('left', 'Z abs', units='Rayl')
    #     elif text == "传输阻抗率Z Re":
    #         self.plot3.setLabel('left', 'Z Re', units='Rayl')
    #     elif text == "传输阻抗率Z Im":
    #         self.plot3.setLabel('left', 'Z Im', units='Rayl')
    #     else:
    #         print("没这个选择项")
    #         return
    #
    #     if not self.test_result:
    #         QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
    #         return
    #     self.plot3.clear()
    #
    #     # 十字线和提示框初始化但不添加到图上
    #     self.vLine = pyqtgraph.InfiniteLine(angle=90, movable=False, pen='r')
    #     self.hLine = pyqtgraph.InfiniteLine(angle=0, movable=False, pen='r')
    #     self.text = pyqtgraph.TextItem("", anchor=(0, 1), fill=pyqtgraph.mkBrush(255, 255, 255, 200), border='k')
    #
    #     # 十字线显示开关
    #     self.crosshair_enabled = False
    #     self.selected_point_index = None
    #
    #     # 绑定双击事件
    #     try:
    #         self.plot3.scene().sigMouseClicked.disconnect(self.on_plot3_clicked)
    #     except Exception:
    #         pass
    #     self.plot3.scene().sigMouseClicked.connect(self.on_plot3_clicked)
    #
    #     if text == "传输阻抗率Z abs":
    #         log_f = np.log10(np.asarray(self.test_result["f"], dtype=float))
    #         log_Z_abs = np.asarray(self.test_result["Z_abs"], dtype=float).copy()
    #         log_Z_abs[log_Z_abs <= 0] = np.nan
    #         log_Z_abs = np.log10(log_Z_abs)
    #         if self.soft_value > 3:
    #             log_Z_abs = savgol_filter(log_Z_abs, window_length=self.soft_value, polyorder=3)
    #         self.plot3.plot(log_f, log_Z_abs, pen=mkPen(color='black', width=2))
    #
    #     elif text == "传输阻抗率Z Re":
    #         log_f = np.log10(np.asarray(self.test_result["f"], dtype=float))
    #         log_Z_Re = np.abs(np.asarray(self.test_result["Z_Re"], dtype=float))
    #         log_Z_Re[log_Z_Re <= 0] = np.nan
    #         log_Z_Re = np.log10(log_Z_Re)
    #         if self.soft_value > 3:
    #             log_Z_Re = savgol_filter(log_Z_Re, window_length=self.soft_value, polyorder=3)
    #         self.plot3.plot(log_f, log_Z_Re, pen=mkPen(color='black', width=2))
    #
    #     elif text == "传输阻抗率Z Im":
    #         log_f = np.log10(np.asarray(self.test_result["f"], dtype=float))
    #         log_Z_Im = np.abs(np.asarray(self.test_result["Z_Im"], dtype=float))
    #         log_Z_Im[log_Z_Im <= 0] = np.nan
    #         log_Z_Im = np.log10(log_Z_Im)
    #         if self.soft_value > 3:
    #             log_Z_Im = savgol_filter(log_Z_Im, window_length=self.soft_value, polyorder=3)
    #         self.plot3.plot(log_f, log_Z_Im, pen=mkPen(color='black', width=2))
    #     else:
    #         print("无")
    #
    #     self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
    #     # self.plot3.plotItem.enableAutoRange(axis='y', enable=False)  # Y轴范围不变
    #     self.plot_state = True

    def del_cross(self):
        # 清竖线
        if hasattr(self, "vLine") and self.vLine is not None:
            self.plot3.removeItem(self.vLine)
            self.vLine = None

        # 清横线
        if hasattr(self, "hLine") and self.hLine is not None:
            self.plot3.removeItem(self.hLine)
            self.hLine = None

        # 清提示文字
        if hasattr(self, "text") and self.text is not None:
            self.plot3.removeItem(self.text)
            self.text = None

        # 清点击的小圆点
        if hasattr(self, "click_marker") and self.click_marker is not None:
            self.plot3.removeItem(self.click_marker)
            self.click_marker = None
        # 标记：当前没有十字线了
        self.crosshair_enabled = False
        # 没有十字线时，也不保留当前选中点
        self.selected_point_index = None

    def popup_pdf(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pdf_path = os.path.join(base_dir, "resources", "file", "小孔膜片阻抗管测试系统操作使用手册.pdf")
        if os.path.exists(pdf_path):
            if sys.platform.startswith("win"):
                os.startfile(pdf_path)  # Windows
            else:
                subprocess.run(["open", pdf_path])  # macOS 用 open；Linux 可用 xdg-open
        else:
            print("❌ PDF 文件不存在！")


    def update_history_lines(self):
        # 读取配置
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            self.logger.error("读取 tree_config.json 失败")
            return

        line = content.get("line", {})
        if line["is_display_history_line"] == 0:
            self.logger.info("历史线显示总开关为 0，跳过绘制。")
            return

        # 取出 state 为 True 的历史线，data_file 和显示信息放在同一个配置项里。
        id_list = []
        for k, v in line["manage_history_line"].items():
            state_val = v.get("state", False)
            line_name = v.get("line_name")
            if state_val == "True":
                try:
                    id_list.append((int(k), v.get("colour"), line_name, v.get("data_file")))
                except Exception:
                    self.logger.warning(f"manage_history_line 的键无法转 int: {k}")
        id_list.sort()
        self.logger.info(f"启用的历史线序号：{id_list}")
        for i, colour, line_name, data_file in id_list:
            original_data = self.load_history_line_data(data_file)
            if original_data is None:
                self.logger.warning(f"历史线 {i} 数据读取失败，已跳过。")
                continue
            self.add_history_line(colour, original_data, line_name)

    def load_history_line_data(self, data_file):
        """
        读取一条历史曲线的数据。
        data_file 是保存在 manage_history_line 单条历史线配置里的 npz 相对路径。
        """
        if not data_file:
            return None

        # 已经加载过的 npz 直接从内存取，减少磁盘 IO。
        cached = self._history_line_data_cache.get(data_file)
        if cached is not None:
            return cached

        data = utils.load_history_line_data("tree_config.json", data_file)
        if data is not None:
            self._history_line_data_cache[data_file] = data
        return data

    def add_history_line(self, colour, original_data, line_name):
        text = self.plot_type_selector.currentText()

        log_f = np.log10(original_data["f"])

        if "abs" in text:
            y = np.asarray(original_data["Z_abs"], dtype=float)
        elif "Re" in text:
            y = np.asarray(original_data["Z_Re"], dtype=float)
        elif "Im" in text:
            y = np.asarray(original_data["Z_Im"], dtype=float)
        else:
            QMessageBox.warning(self, "提示", f"无{text}选项")
            return

        if self.soft_value > 3:
            y = savgol_filter(y, window_length=self.soft_value, polyorder=3)

        # 这里新增：直接画、并记录到 history_line，方便下次 removeItem
        item = self.plot3.plot(log_f, y, pen=mkPen(color=colour, width=1), name=line_name)
        self.history_line.append(item)

    def _get_impedance_tube_parameter_inputs(self):
        if not self.tube_params:
            QMessageBox.warning(self, "参数错误", "阻抗管参数未加载，请先检查参数设置。")
            return None
        try:
            params = {
                "nfft": int(self.tube_params.get("nfft")),
                "sample_area_mm2": float(self.tube_params.get("sample_area_mm2")),
                "incident_tube_radius_mm": float(self.tube_params.get("incident_tube_radius_mm")),
                "rear_mic_sensitivity_v_per_pa": float(self.tube_params.get("rear_mic_sensitivity_v_per_pa")),
                "back_cavity_radius_mm": float(self.tube_params.get("back_cavity_radius_mm")),
                "front_mic_distance_mm": float(self.tube_params.get("front_mic_distance_mm")),
                "back_cavity_depth_mm": float(self.tube_params.get("back_cavity_depth_mm")),
                "temp": float(self.tube_params.get("tube_temperature")),
                "atmospheric_pressure_kpa": float(self.tube_params.get("atmospheric_pressure_kpa")),
            }
        except (TypeError, ValueError):
            QMessageBox.warning(self, "参数错误", "阻抗管参数不完整，请打开“阻抗管参数设置”补全后保存。")
            return None

        if params["nfft"] <= 1:
            QMessageBox.warning(self, "参数错误", "FFT 点数必须大于 1。")
            return None
        positive_keys = (
            "sample_area_mm2", "incident_tube_radius_mm", "rear_mic_sensitivity_v_per_pa", "back_cavity_radius_mm",
            "front_mic_distance_mm", "back_cavity_depth_mm", "atmospheric_pressure_kpa",
        )
        if any(params[key] <= 0 for key in positive_keys):
            QMessageBox.warning(self, "参数错误", "样品面积、几何尺寸和大气压必须大于 0。")
            return None

        return params
