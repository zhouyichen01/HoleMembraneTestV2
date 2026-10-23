import sounddevice as sd
from PyQt5 import QtWidgets, sip
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QDialog, QMessageBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, \
    QHeaderView
from PyQt5.uic import loadUi
from control.log_manager import LogManager
from control.utils import utils


class SelectDeviceInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.signal_info = None
        # 缩放比例
        self._ui_scale = 1.0
        self.logger = LogManager.set_log_handler("硬件设备选择")
        self.init_ui()
        self.init_fun()
        self.device_list = []
        self.select_devices()

    def init_ui(self):
        ui_file = QFile(":ui/select_device.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 select_device.ui")
            raise FileNotFoundError("未找到资源文件 select_device.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("硬件设备选择")
        # 最小化
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        # 先按屏幕可用区域缩小窗口
        utils.resize_by_ui_with_screen(self)
        self.show()
        self.logger.info("打开硬件设备选择界面")

    def _scaled(self, value, minimum):
        scaled_value = utils.scale_ui_value(self, value, minimum)
        return scaled_value

    def init_fun(self):
        self.save_all_button.clicked.connect(self.save_all_device_settings)
        self.refresh_devices.clicked.connect(self.refresh_device_list)
        self.comboBox.setMinimumWidth(self._scaled(350, 180))
        self.comboBox_2.setMinimumWidth(self._scaled(350, 180))
        # 加永久布局
        self.groupBox_4 = self.findChild(QGroupBox, 'groupBox_4')
        self.channels_layout = QVBoxLayout(self.groupBox_4)
        self.groupBox_4.setLayout(self.channels_layout)
        self.groupBox_4.setFixedWidth(self._scaled(140, 90))
        self.groupBox_5.setFixedWidth(self._scaled(180, 130))
        self.vp_line_edit.setFixedWidth(self._scaled(70, 55))
        self.save_vp_button.setFixedWidth(self._scaled(75, 65))
        self.save_vp_button.clicked.connect(self.save_vp_value)
        self.load_vp_value()

    def load_vp_value(self):
        res, config = utils.get_config_content("basic_params.json")
        if res:
            vp_value = config.get("basic_params", {}).get("Vp", "")
            self.vp_line_edit.setText(str(vp_value))

    def save_vp_value(self):
        vp_text = self.vp_line_edit.text().strip()
        if not vp_text:
            QMessageBox.warning(self, "警告", "请输入Vp值")
            return

        try:
            vp_value = float(vp_text)
        except ValueError:
            QMessageBox.warning(self, "警告", "Vp值必须是数字")
            return

        res, config = utils.get_config_content("basic_params.json")
        if not res:
            QMessageBox.critical(self, "保存失败", "无法读取配置文件内容 basic_params.json！")
            return

        config.setdefault("basic_params", {})["Vp"] = vp_value
        if utils.write_config_content("basic_params.json", config):
            QMessageBox.information(self, "保存成功", "Vp值已保存！")
        else:
            QMessageBox.critical(self, "保存失败", "配置文件保存失败！")

    def refresh_device_list(self):
        # 刷新设备列表
        self.logger.info("正在刷新设备列表...并初始化设备...")
        sd._terminate()  # 终止当前的设备实例
        sd._initialize()  # 重新初始化设备
        utils.init_selected_devide()
        self.select_devices()  # 重新显示设备
        self.logger.info("设备列表刷新完毕")

    def select_devices(self):
        self.device_list = sd.query_devices()
        hostapis = sd.query_hostapis()
        self.logger.info(f"默认设备：out{sd.default.device[1]}, in{sd.default.device[0]}")

        # 输出设备
        self.comboBox.clear()
        self.comboBox.addItem("请选择输出设备")
        hint_index = self.comboBox.count() - 1
        hint_item = self.comboBox.model().item(hint_index)
        hint_item.setEnabled(False)  # 禁用点击
        hint_item.setForeground(Qt.gray)  # 设置灰色文字
        # 输入设备
        self.comboBox_2.clear()
        self.comboBox_2.addItem("请选择输入设备")
        hint_index2 = self.comboBox_2.count() - 1
        hint_item2 = self.comboBox_2.model().item(hint_index2)
        hint_item2.setEnabled(False)  # 禁用点击
        hint_item2.setForeground(Qt.gray)  # 设置灰色文字

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["序号", "名称", "输入通道", "输出通道", "驱动类型", "采样率"])

        for index, device in enumerate(self.device_list):
            hostapi_index = device['hostapi']
            hostapi_name = hostapis[hostapi_index]['name']
            display_name = f"{device['name']} ({hostapi_name})"
            display_name_with_channels = (f'{display_name} [OUT: {device["max_output_channels"]}, '
                                          f'IN: {device["max_input_channels"]}]')

            samplerate = device.get("default_samplerate", "N/A")
            if isinstance(samplerate, float):
                samplerate = f"{int(samplerate)} Hz"

            # 表格显示所有设备
            row_items = [
                QStandardItem(str(index)),
                QStandardItem(display_name),
                QStandardItem(str(device["max_input_channels"])),
                QStandardItem(str(device["max_output_channels"])),
                QStandardItem(hostapi_name),
                QStandardItem(str(samplerate))
            ]
            # 给名称列设置 tooltip
            row_items[1].setToolTip(display_name)
            row_items[4].setToolTip(hostapi_name)

            for item in row_items:
                item.setEditable(False)
                item.setTextAlignment(Qt.AlignCenter)
            model.appendRow(row_items)

            # 输出设备填充
            if device["max_output_channels"] > 0:
                self.comboBox.addItem(f"{index}: {display_name_with_channels}", index)

            # 输入设备填充
            if device["max_input_channels"] > 0:
                self.comboBox_2.addItem(f"{index}: {display_name_with_channels}", index)

        found_out = False
        for i in range(1, self.comboBox.count()): # 从 1 开始，跳过提示项
            if self.comboBox.itemData(i) == sd.default.device[1]:
                self.comboBox.setCurrentIndex(i)
                found_out = True
                break
        if not found_out:
            self.comboBox.setCurrentIndex(0)
        found_in = False
        for i in range(1, self.comboBox_2.count()):
            if self.comboBox_2.itemData(i) == sd.default.device[0]:
                self.comboBox_2.setCurrentIndex(i)
                found_in = True
                break
        if not found_in:
            self.comboBox_2.setCurrentIndex(0)

        self.tableView.setModel(model)
        self._apply_table_sizing()
        # 自动加载通道
        self.load_channels()

    def load_channels(self):
        res, config = utils.get_config_content("basic_params.json")
        if res:
            channel_count = config["basic_params"].get("input_channel_count", 0)
            # 每次加载前清空之前的通道ui
            self.clear_layout(self.channels_layout)
            self.add_channels_to_ui(channel_count)
        else:
            QMessageBox.critical(self, "读取失败", "无法读取配置文件内容 basic_params.json！")

    def add_channels_to_ui(self, channel_count):
        # 读取 mic_calibration.json 文件，获取每个通道的 binding_index
        res, mic_config = utils.get_config_content("mic_calibration.json")
        if res:
            for i in range(1, channel_count + 1):
                channel_name = f"In{i}"

                row_layout = QHBoxLayout()
                # 左侧显示通道名
                channel_label = QLabel(f"Mic{i}:", self)
                row_layout.addWidget(channel_label)
                # 右侧显示对应的 QComboBox
                combo_box = QComboBox(self)
                combo_box.setObjectName(f"comboxBox_In{i}")
                combo_box.setFixedWidth(self._scaled(50, 42))  # 设置宽度

                # 将 In1, In2, In3, In4 等按顺序添加到下拉框
                for j in range(1, channel_count + 1):
                    combo_box.addItem(f"In{j}")
                # 获取该通道的 binding_index
                binding_index = mic_config.get(channel_name, {}).get("binding_index", -1)
                if binding_index != -1:
                    combo_box.setCurrentIndex(binding_index)
                row_layout.addWidget(combo_box)
                self.channels_layout.addLayout(row_layout)
            # 添加保存按钮
            save_channel_button = QPushButton("保存通道设置", self.groupBox_4)
            save_channel_button.setFixedWidth(self._scaled(120, 80))
            save_channel_button.clicked.connect(lambda: self.save_channel_settings(channel_count))
            self.channels_layout.addWidget(save_channel_button, alignment=Qt.AlignCenter)

            # 轻量刷新
            self.groupBox_4.updateGeometry()
            self.groupBox_4.update()
        else:
            QMessageBox.critical(self, "读取失败", "无法读取配置文件内容 mic_calibration.json！")

    def clear_layout(self, layout):
        """
        清空 选择通道groupBox_4下的ui
        """
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_layout(child_layout)
                    # 删除子布局自身，避免泄漏
                    # PyQt里没有 deleteLater() 给 QLayout，用 sip.delete
                    try:
                        sip.delete(child_layout)
                    except Exception:
                        pass

    def save_all_device_settings(self):
        output_index = self.comboBox.currentData()
        input_index = self.comboBox_2.currentData()

        if input_index is None or output_index is None:
            QMessageBox.warning(self, "警告", "请选择输入和输出设备")
            return

        try:
            sd.default.device = (input_index, output_index)
            self.logger.info(f"设置默认输入设备为: {self.device_list[input_index]['name']}")
            self.logger.info(f"设置默认输出设备为: {self.device_list[output_index]['name']}")

            # 保存驱动信息至config
            hostapis = sd.query_hostapis()
            input_device = self.device_list[input_index]
            output_device = self.device_list[output_index]

            input_device_index = input_device['index']
            output_device_index = output_device['index']

            input_device_name = input_device['name']
            output_device_name = output_device['name']

            input_type = hostapis[input_device['hostapi']]['name']
            output_type = hostapis[output_device['hostapi']]['name']

            input_samplerate = str(int(input_device.get('default_samplerate', 0)))
            output_samplerate = str(int(output_device.get('default_samplerate', 0)))

            # 获取输入设备的通道数
            input_channel_count = input_device.get('max_input_channels', 0)

            utils.save_selected_devices(input_device_index, output_device_index, input_device_name, output_device_name,
                                        input_type, output_type, input_samplerate, output_samplerate,
                                        input_channel_count)
            # 重载json
            utils.refresh_mic_config_json(self)
            QMessageBox.information(self, "成功", "输入输出设备保存成功！")
            # 刷新选择通道ui
            self.load_channels()
        except Exception as e:
            self.logger.error(f"设置默认输入输出设备失败: {e}")
            QMessageBox.critical(self, "错误", f"设置默认输入输出设备失败: {e}")

    def save_channel_settings(self, channel_count):
        res, config = utils.get_config_content("mic_calibration.json")
        if res:
            try:
                for i in range(1, channel_count + 1):
                    combo_box = self.findChild(QComboBox, f"comboxBox_In{i}")
                    if combo_box:
                        binding_index = combo_box.currentIndex()  # 获取当前选择的 binding_index
                        channel_name = f"In{i}"
                        if channel_name in config:
                            config[channel_name]["binding_index"] = binding_index
                        else:
                            raise KeyError(f"{channel_name}该通道不存在！")

                res = utils.write_config_content("mic_calibration.json", config)
                if res:
                    QMessageBox.information(self, "保存成功", "通道设置已成功保存！")
                else:
                    QMessageBox.critical(self, "保存失败", "配置文件保存失败！")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存通道设置失败: {e}")
                self.logger.error(f"保存通道设置失败: {e}")
        else:
            QMessageBox.critical(self, "读取失败", "无法读取配置文件内容 mic_calibration.json！")

    def _apply_table_sizing(self):
        """按比例设置列宽，名称列吃掉剩余空间"""
        header = self.tableView.horizontalHeader()
        header.setStretchLastSection(False)

        # 各列固定宽度（可以根据你截图里的效果调整数值）
        self.tableView.setColumnWidth(0, self._scaled(40, 32))  # 序号
        self.tableView.setColumnWidth(2, self._scaled(60, 45))  # 输入通道
        self.tableView.setColumnWidth(3, self._scaled(60, 45))  # 输出通道
        self.tableView.setColumnWidth(4, self._scaled(180, 100))  # 驱动类型
        self.tableView.setColumnWidth(5, self._scaled(100, 78))  # 采样率

        header.setSectionResizeMode(1, QHeaderView.Stretch)  # 名称填充

        self.tableView.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)  # 居中
        self.tableView.verticalHeader().setVisible(False)  # 行号不可见
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  # 整行高亮

    def closeEvent(self, event):
        if self.parent:
            self.parent.device_window = None
        self.logger.info("关闭硬件设备选择界面")
        event.accept()
