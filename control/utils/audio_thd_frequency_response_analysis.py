from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter1d
from scipy.signal import hilbert


class AudioThdFrequencyResponseAnalysis(object):
    @staticmethod
    def spl_calculation(recorded_signal,
                        reference_pressure: float = 20e-6,
                        window_size: int = 1201,
                        method: str = "rms",
                        padding_mode: str = "zero",
                        padding_cval: Optional[float] = 0.0,
                        v2pa_factor: Optional[float] = None,
                        trim_edges: bool = False,
                        vp = 0
                        ):
        """
            使用滑动窗口计算声压级 SPL（dB）。

            - method: "rms" | "envelope" | "hilbert"，默认使用 "rms"
            - padding_mode: "reflect" | "nearest" | "mirror" | "wrap" | "constant"
              支持别名 "zero" -> "constant"
            - padding_cval: 仅当 padding_mode == "constant" 时使用
        """
        method = str(method).strip().lower()
        padding_mode_in = str(padding_mode).strip().lower()

        alias_to_mode = {
            "zero": "constant",
            "zeros": "constant",
            "const": "constant",
            "constant": "constant",
        }
        valid_modes = {"reflect", "nearest", "mirror", "wrap", "constant"}
        mode = alias_to_mode.get(padding_mode_in, padding_mode_in)
        if mode not in valid_modes:
            mode = "constant"

        valid_methods = {"rms", "envelope", "hilbert"}
        if method not in valid_methods:
            method = "rms"

        if mode == "constant":
            cval = 0.0 if padding_cval is None else float(padding_cval)
        else:
            cval = 0.0

        signal_float = np.asarray(recorded_signal, dtype=float)
        if method == "envelope":
            v_amp_list = maximum_filter(np.abs(signal_float), size=window_size, mode=mode, cval=cval)
        elif method == "hilbert":
            amplitude_envelope = np.abs(hilbert(signal_float))
            v_amp_list = uniform_filter1d(amplitude_envelope, size=window_size, axis=0, mode=mode, cval=cval)
        else:
            v_amp_list = np.sqrt(uniform_filter1d(signal_float**2, size=window_size, axis=0, mode=mode, cval=cval))
        if vp > 0:
            dbfs = np.mean(20 * np.log10(np.abs(v_amp_list) * vp))
        else:
            dbfs = None
        if not v2pa_factor:
            v2pa_factor = 1
        pa_amp_list = np.asarray(v_amp_list, dtype=float) * v2pa_factor
        pa_amp_list = np.maximum(pa_amp_list, 1.0e-10)  # Avoid log of zero or negative due to numerical issues
        spl = 20 * np.log10(pa_amp_list / float(reference_pressure))

        if not trim_edges:
            return spl, dbfs

        n = int(signal_float.size)
        w = int(window_size)
        if w <= 1 or w > n:
            return spl, dbfs
        out_len = n - w + 1
        start = w // 2
        end = start + out_len
        return spl[start:end], dbfs
