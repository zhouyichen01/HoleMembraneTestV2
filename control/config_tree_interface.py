import re

from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QFrame, \
    QToolButton, QSpacerItem, QSizePolicy, QInputDialog
from control.utils import utils
from custom.customSignals import sign
from control.log_manager import LogManager
from PyQt5.QtCore import QFile, Qt
from PyQt5.uic import loadUi



class ConfigTreeInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.logger = LogManager.set_log_handler("历史数据设置")
        self.init_ui()
        self.init_fun()

    def init_ui(self):
        ui_file = QFile(":ui/tree_config.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 tree_config.ui")
            raise FileNotFoundError("未找到资源文件 tree_config.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("历史数据设置")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint| Qt.WindowMaximizeButtonHint)
        # 调整界面大小
        utils.resize_by_ui_with_screen(self)
        self.logger.info("打开历史数据设置界面")

    def init_fun(self):
        # 左侧树 → 切换页面
        self.treeWidget.itemClicked.connect(self.on_tree_click)
        self.buttonGroup.addButton(self.radioButton, 1)  # 置顶=1
        self.buttonGroup.addButton(self.radioButton_2, 0)  # 不置顶=0
        self.buttonGroup.buttonClicked[int].connect(self.on_group_clicked)


    def on_tree_click(self, item):
        text = item.text(0)
        if text == "是否置顶":
            self.ui.stackedWidget.setCurrentIndex(0)
            self.fill_page()
        elif text == "历史线":
            self.ui.stackedWidget.setCurrentIndex(1)
            self.fill_page_2()
        elif text == "历史数据":
            self.ui.stackedWidget.setCurrentIndex(2)

    def fill_page(self):
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            self.logger.error("读取 tree_config.json 失败")
            return
        else:
            val = content.get("line", {}).get("is_display_history_line")
            if val:
                self.radioButton.setChecked(True)
            else:
                self.radioButton_2.setChecked(True)

    def on_group_clicked(self, bid: int):
        # 选中“置顶”为 1，“不置顶”为 0
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            self.logger.error("读取 tree_config.json 失败，无法写入。")
            QMessageBox.warning(self, "错误", f"读取 tree_config.json 失败，无法写入。")
            return

        content["line"]["is_display_history_line"] = bid
        res= utils.write_config_content("tree_config.json", content)
        if res:
            self.logger.info("tree_config.json 置顶写入成功")
            return
        else:
            self.logger.info("tree_config.json 置顶写入失败")
            QMessageBox.warning(self, "错误", f"tree_config.json 写入失败")
            return

    def fill_page_2(self):
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            self.logger.error("读取 tree_config.json 失败")
            return

        need_display = content.get("line", {}).get("manage_history_line", {})
        # 无数据时,加 "暂无历史线" 提示
        if not need_display:
            if self.ui.groupBox.layout() is not None:
                QWidget().setLayout(self.ui.groupBox.layout())

            layout = QVBoxLayout(self.ui.groupBox)
            layout.setContentsMargins(0, 0, 0, 0)

            tip = QLabel("暂无历史线")
            tip.setAlignment(Qt.AlignCenter)
            tip.setStyleSheet("""
                QLabel {
                    color: #9AA0A6;
                    font-size: 9pt;
                    padding: 8px;
                }
            """)
            tip.setWordWrap(True)

            layout.addStretch()
            layout.addWidget(tip)
            layout.addStretch()

            self.ui.groupBox.setLayout(layout)
            return

        # 清理旧布局（如果重新进入）
        if self.ui.groupBox.layout() is not None:
            QWidget().setLayout(self.ui.groupBox.layout())
        layout = QVBoxLayout(self.ui.groupBox)

        # 动态生成行
        for key, i in need_display.items():
            state = str(i.get("state", "False")) == "True"
            colour = i.get("colour", "white")
            line_name = i.get("line_name")
            if not line_name:  # 处理 None、""、空格、False 等情况
                line_name = f"历史线{key}"
            # 每一行一个水平布局
            row = QHBoxLayout()

            # 复选框
            cb = QCheckBox()
            cb.setChecked(state)
            cb.stateChanged.connect(lambda state, index=key: self.on_history_line_toggle(index, state))

            # 彩色线条
            color_line = QFrame()
            color_line.setFixedHeight(2)
            color_line.setFixedSize(20, 20)
            color_line.setFrameShape(QFrame.HLine)
            color_line.setStyleSheet(f"background-color: {colour}; border: none;")

            label_name = QLabel(f"{line_name}: ")
            label_name.setFixedWidth(120)
            label_name.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            # 重命名按钮
            btn_rename = QToolButton()
            btn_rename.setText("重命名")
            btn_rename.setFixedWidth(60)
            btn_rename.setStyleSheet("""
                          QToolButton {
                              padding-top: 6px;
                              padding-bottom: 3px;
                          }
                      """)
            btn_rename.clicked.connect(
                lambda _, index=key, name_object=label_name: self.on_history_line_rename(index, name_object))

            # 删除按钮
            btn_delete = QToolButton()
            btn_delete.setText("删除")
            btn_delete.setFixedWidth(50)
            btn_delete.setStyleSheet("""
                QToolButton {
                padding-top: 6px;
                padding-bottom: 3px;
                }
            """)
            btn_delete.clicked.connect(lambda _, index=key, name=line_name: self.delete_history_line(index, name))

            # 伸缩弹簧ui
            horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

            row.addWidget(cb)
            row.addWidget(label_name)
            row.addWidget(color_line)
            row.addWidget(btn_rename)
            row.addWidget(btn_delete)
            row.addItem(horizontalSpacer)
            layout.addLayout(row)

        layout.addStretch()
        self.ui.groupBox.setLayout(layout)
        self.logger.info("填充 page_2 完成")

    def on_history_line_rename(self, index, name_object):
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            QMessageBox.warning(self, "错误", "读取配置失败，无法重命名。")
            return

        current = name_object.text()
        new_name, ok = QInputDialog.getText(self, "重命名", "名称：", text=current)
        if not ok:
            return
        new_name = (new_name or "").strip()
        if not new_name:
            QMessageBox.warning(self, "格式错误", "名称不能为空。")
            return
        if len(new_name) > 12:
            QMessageBox.warning(self, "格式错误", "名称长度不能超过 12 个字符。")
            return
        if not re.match(r'^[A-Za-z0-9 _\-\(\):]+$', new_name):
            QMessageBox.warning(self, "格式错误", "仅允许：英文、数字、空格、_、-、()、:")
            return

        try:
            content["line"]["manage_history_line"][str(index)]["line_name"] = new_name
            ok = utils.write_config_content("tree_config.json", content)
            if ok:
                name_object.setText(new_name)  # 更新 UI
                QMessageBox.information(self, "成功", f"已重命名为：{new_name}")
                sign.update_plot3_by_selector_sign.emit("config_tree")  # 刷新主窗 legend
            else:
                QMessageBox.warning(self, "错误", "保存名称失败。")
        except Exception as e:
            self.logger.error(f"写入名称失败: {e}")
            QMessageBox.critical(self, "错误", f"写入名称失败：{e}")


    def on_history_line_toggle(self, index, state):
        """
        用户勾选/取消复选框时，实时写入配置文件。
        """
        res, content = utils.get_config_content("tree_config.json")
        if not res:
            self.logger.error("读取 tree_config.json 失败（on_history_line_toggle）")
            QMessageBox.warning(self, "错误", "读取配置失败，无法写入。")
            return

        state_str = "True" if state else "False"
        try:
            content["line"]["manage_history_line"][str(index)]["state"] = state_str
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
            return

        res = utils.write_config_content("tree_config.json", content)
        if res:
            self.logger.info(f"更新历史线 {index} 状态成功 → {state}")
        else:
            self.logger.error(f"更新历史线 {index} 状态失败")
            QMessageBox.warning(self, "错误", f"更新历史线 {index} 状态失败")

    def delete_history_line(self, key, name):
        """
        删除一条历史线并更新配置文件。
        """
        user_choice = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除“{name}”吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if user_choice != QMessageBox.Yes:
            return
        res, content = utils.get_config_content("tree_config.json")

        if not res:
            QMessageBox.warning(self, "错误", "读取配置失败，无法删除。")
            return

        # data_file 和显示信息放在同一个历史线配置项里，删除前先取出对应 npz 路径。
        delete_entry = content["line"]["manage_history_line"][str(key)]
        data_file = delete_entry.get("data_file")

        # 先删除配置里的显示项，等配置写入成功后再删实际 npz 文件。
        del content["line"]["manage_history_line"][str(key)]

        old_lines = content["line"]["manage_history_line"]
        new_lines = {}
        for index, val in old_lines.items():
            print(index,val)
            ik = int(index)
            if ik < int(key):
                new_lines[index] = val  # 保持原键不动
            elif ik > int(key):
                new_lines[str(ik - 1)] = val  # 仅对后面的 -1

        # 删除中间某条后重新编号，保证下一次新增历史线仍能按连续序号写入。
        content["line"]["manage_history_line"] = new_lines

        res = utils.write_config_content("tree_config.json", content)
        if res:
            if data_file:
                # 配置已经成功写入，此时可以清理不再被引用的历史曲线数据文件。
                utils.delete_history_line_data("tree_config.json", data_file)
            self.logger.info(f"已删除历史线 {key}")
            QMessageBox.information(self, "提示", f"{name} 已删除。")
            self.fill_page_2()
            # 通知主窗口刷新 plot3（当前曲线+历史线）
            sign.update_plot3_by_selector_sign.emit("config_tree")
        else:
            # 写入失败时只提示用户，磁盘上的 npz 数据文件会保留。
            self.logger.error(f"删除历史线 {key} 写入配置失败")
            QMessageBox.warning(self, "错误", f"{name} 删除失败，配置文件写入失败。")
