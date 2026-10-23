import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class MySignals(QObject):
    """
    自定义信号类型
    """
    play_audio_sign = pyqtSignal()
    play_adjust_audio_sign = pyqtSignal()
    error_message_signal = pyqtSignal(str, object)
    success_message_signal = pyqtSignal(str, object)
    warning_message_signal = pyqtSignal(str, object)
    update_plot_sign = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, int)
    update_plot3_by_selector_sign = pyqtSignal(str)


sign = MySignals()


