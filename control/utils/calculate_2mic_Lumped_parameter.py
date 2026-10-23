def calculate_impedance_Lumped_parameter(
    mic2, mic1, mic2_cal, mic1_cal, sf, temp, s_sample_mm2, v_backing_cc
):
    """
    基于集总参数法（背腔阻抗 Za）计算样品的比流阻 Z。

    参数:
    - mic2, mic1, mic2_cal, mic1_cal : ndarray
        测量与校准四路传声器采集的时域数据（numpy 数组，长度可不同，函数内自动对齐截断）。
    - sf : int
        采样率（Hz）
    - temp : float
        实验温度（单位: 摄氏度）
    - s_sample_mm2 : float
        样品有效面积（单位: mm^2）
    - v_backing_cc : float
        背腔体积（单位: cc，即 mL）

    返回:
    - f_subset : ndarray
        频率向量（10~10000 Hz）
    - Z_abs, Z_Re, Z_Im : ndarray
        Z 的模、实部、虚部（单位: Pa*s/m）
    """
    import numpy as np
    from scipy.signal import csd, welch

    # 常量
    temp0 = 293
    atm0 = 101.325
    rho0 = 1.186
    atm = 101

    # 计算声速、密度
    c0 = 343.2 * np.sqrt((temp + 273) / temp0)
    rho = rho0 * (atm / atm0) * (temp0 / (temp + 273))

    # 对齐四路信号长度（取最短长度截断）
    min_len = min(len(mic2), len(mic1), len(mic2_cal), len(mic1_cal))
    if min_len < 2:
        raise ValueError("Signals must contain at least two samples.")
    mic2 = np.asarray(mic2, dtype=float).reshape(-1)[:min_len]
    mic1 = np.asarray(mic1, dtype=float).reshape(-1)[:min_len]
    mic2_cal = np.asarray(mic2_cal, dtype=float).reshape(-1)[:min_len]
    mic1_cal = np.asarray(mic1_cal, dtype=float).reshape(-1)[:min_len]

    # 频率参数
    nfft = sf
    w = np.hanning(nfft)
    noverlap = nfft // 2

    # 传递函数估计（等效 MATLAB tfestimate(input, output): output / input）
    def tfestimate(x, y):
        f, Pxy = csd(
            x, y, fs=sf, window=w, nperseg=nfft, noverlap=noverlap, nfft=nfft
        )
        _, Pxx = welch(
            x, fs=sf, window=w, nperseg=nfft, noverlap=noverlap, nfft=nfft
        )
        return Pxy / Pxx, f

    h21_test, ff = tfestimate(mic2, mic1)
    h21_cal, _ = tfestimate(mic2_cal, mic1_cal)

    # 有效点筛选
    valid = (
        (ff > 0)
        & np.isfinite(ff)
        & np.isfinite(h21_test)
        & np.isfinite(h21_cal)
        & (np.abs(h21_cal) > 0)
    )
    ff = ff[valid]
    h21 = h21_test[valid] / h21_cal[valid]

    # 单位换算
    s_sample = s_sample_mm2 * 1e-6  # mm^2 -> m^2
    v_backing = v_backing_cc * 1e-6  # cc -> m^3

    # 集总参数法阻抗：背腔声顺 Ca 与背腔阻抗 Za
    ca = v_backing / (rho * c0**2)
    za = -1j / ((2 * np.pi * ca) * ff)
    Z = (h21 - 1) * za * s_sample

    # 频率范围筛选
    ind = (ff >= 10) & (ff <= 10000)
    f_subset = ff[ind]
    Z_subset = Z[ind]

    # 模值和实虚部
    Z_abs = np.abs(Z_subset)
    Z_Re = np.real(Z_subset)
    Z_Im = np.imag(Z_subset)

    return f_subset, Z_abs, Z_Re, Z_Im
