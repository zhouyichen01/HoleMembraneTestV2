import queue

import numpy as np
import sounddevice as sd

from control.log_manager import LogManager

"""
Streaming audio processor using sounddevice callbacks for real-time recording.
Enables non-blocking audio capture with real-time chunk processing.


本模块实现一个全双工（播放+录音）的音频流：
- 使用 sounddevice 的 callback 机制进行实时音频处理
- 播放预先给定的 stimulus，同时同步录制输入信号
- 通过队列实现非阻塞的数据采集
"""

class DuplexStreamingPlayRec:
    def __init__(self):
        # sounddevice Stream 对象
        self.stream = None
        # 回调中采集到的音频数据队列（线程安全）
        self.audio_queue = queue.Queue()
        # 已经累计的所有音频块
        self.accumulated_chunks = []
        # 当前是否处于录音状态
        self.is_recording = False
        # 采样率（Hz）
        self.sample_rate = 44100
        # 目标采样点数（达到即停止）
        self.target_samples = 0
        # 当前已采集的采样点数
        self.samples_captured = 0
        # 回调或流中出现的异常
        self.error = None
        # ===== 播放相关 =====
        # 播放缓冲区，形状：(total_frames, out_channels)
        self._playback_data = None
        # 当前播放到的位置（帧索引）
        self._playback_index = 0
        self.logger = LogManager.set_log_handler("Stream")


    def start(self, stimulus_data, sample_rate, input_device, output_device,
              input_channels=4, output_channels=1, duration=None, blocksize=2048):
        """
        启动全双工音频流（边播边录）
        参数说明：
        - stimulus_data : 要播放的音频数据（numpy array）
        - sample_rate : 采样率
        - input_device : 输入设备索引
        - output_device : 输出设备索引
        - input_channels: 输入通道数
        - output_channels: 输出通道数
        - target_samples: 目标采样点数（优先级高于 duration）
        - duration : 录制时长（秒）
        - blocksize : 每次回调处理的帧数
        """
        # 1) 计算目标采样点数
        if not duration:
            raise ValueError("必须提供 target_samples 或 duration")
        target_samples = int(duration * sample_rate)

        # 2) 强制刷新 PortAudio（解决设备切换/异常状态问题）
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            # 某些版本或状态下可能失败，忽略即可
            self.logger.error(f"强制刷新 PortAudio失败: {e}")
            pass

        # 3) 固定输入/输出两个设备
        input_device = int(input_device)
        output_device = int(output_device)
        sd.default.device = (input_device, output_device)

        # 初始化内部状态
        self.sample_rate = int(sample_rate)
        self.target_samples = target_samples
        self.samples_captured = 0
        self.accumulated_chunks = []
        self.error = None
        self.is_recording = True

        # # 4) stimulus 数据统一成二维数组 (N, out_channels)
        # stimulus_data = np.asarray(stimulus_data, dtype=np.float32)
        # if stimulus_data.ndim == 1:
        #     # 单通道数据转为 (N, 1)
        #     stimulus_data = stimulus_data.reshape(-1, 1)
        # if stimulus_data.shape[1] != output_channels:
        #     # 如果通道数不匹配，则复制第一个通道到所有输出通道
        #     stimulus_data = np.tile(stimulus_data[:, [0]], (1, output_channels))


        # 5) 确保播放数据长度 >= target_samples，不足则补 0
        if stimulus_data.shape[0] < target_samples:
            pad = np.zeros((target_samples - stimulus_data.shape[0], output_channels), dtype=np.float32)
            self._playback_data = np.vstack([stimulus_data, pad])
        else:
            self._playback_data = stimulus_data[:target_samples, :]
        # 播放索引复位
        self._playback_index = 0

        def deal_playrec_block(indata, outdata, frames, time_info, status):
            """
            sounddevice 的实时回调函数：
            - outdata: 写入要播放的音频
            - indata : 读取录制到的音频
            """
            try:
                # ===== 播放部分 =====
                end = self._playback_index + frames
                # 默认填充为 0（静音）
                outdata[:] = 0
                # 如果还有播放数据，则拷贝到 outdata
                if self._playback_index < self._playback_data.shape[0]:
                    chunk = self._playback_data[self._playback_index:min(end, self._playback_data.shape[0]), :]
                    outdata[:chunk.shape[0], :] = chunk
                # 更新播放索引
                self._playback_index += frames

                # ===== 录音部分 =====
                # 拷贝输入数据，避免引用回调缓冲区
                rec = indata.copy().astype(np.float32)  # (frames, in_ch)

                before = self.samples_captured
                self.samples_captured += rec.shape[0]

                # 是否刚好达到目标采样点数
                reached = before < self.target_samples and self.samples_captured >= self.target_samples
                if reached:
                    # 如果超过目标采样点数，裁剪多余部分
                    excess = self.samples_captured - self.target_samples
                    if excess > 0:
                        rec = rec[:-excess, :]
                        self.samples_captured = self.target_samples

                # 将录音数据放入队列（非阻塞）
                try:
                    self.audio_queue.put_nowait(rec)
                except queue.Full:
                    # 队列满则丢弃该块
                    pass

                # 达到目标长度后，异步停止流
                if reached:
                    self.is_recording = False

            except Exception as e:
                # 回调中发生异常时，记录错误并停止流
                self.error = e
                self.logger.error(f"callback error: {e}")
                self.is_recording = False

        # 6) 创建全双工音频流（输入 + 输出）
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=blocksize,
            device=(input_device, output_device),  # 二元组：(输入设备, 输出设备)
            channels=(int(input_channels), int(output_channels)),
            dtype="float32",
            callback=deal_playrec_block,
        )
        # 启动音频流
        self.stream.start()
        self.logger.info("Stream 对象 已开启（流式录制开始）")


    def process_queue(self):
        """
        处理并清空当前队列中的音频块
        返回本次取出的所有块列表
        """
        chunks = []
        while True:
            try:
                c = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            self.accumulated_chunks.append(c)
            chunks.append(c)
        return chunks

    def get_recorded_data(self):
        """
        返回当前已经录制到的全部音频数据（拼接后）
        """
        if not self.accumulated_chunks:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(self.accumulated_chunks)

    def stop(self):
        """
        停止录音与播放，并释放音频流资源
        """
        self.is_recording = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.logger.info(f"Stream 对象 已停止并关闭")
            else:
                self.logger.info("stream=None（无需关闭）")
                return
        except Exception as e:
            self.logger.exception(f"关闭 Stream 异常: {e}")
        finally:
            self.stream = None