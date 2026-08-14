"""
Làm mượt f0. Đây là state của NỐT — vòng đời khác với state của LUỒNG (đuôi FIR).
Trộn hai thứ đó chính là nguyên nhân của jitter 25 cent; xem Config mục C3.
"""

import numpy as np

from tuner.core.config import Config
from tuner.core.music import cents


# =============================================================================
# 5. HẬU XỬ LÝ — median rồi EMA, chạy trên f0 TRƯỚC khi map sang nốt
# =============================================================================

class MedianFilter:
    """Median cửa sổ N: loại outlier đơn lẻ mà không kéo giá trị như trung bình cộng.
    Giá phải trả là trễ nhóm ≈ (N−1)/2 frame.

    LƯU Ý: cửa sổ lớn chỉ có tác dụng nếu bộ đệm được phép ĐẦY. Bản cũ reset mỗi
    khi một frame tụt dưới gate nên median=15 hành xử gần như median=1 — xem
    Config mục C3."""
    def __init__(self, size=15):
        self.size, self.buf = size, []
    def push(self, v):
        self.buf.append(v)
        if len(self.buf) > self.size:
            self.buf.pop(0)
        return float(np.median(self.buf))
    def reset(self):
        self.buf.clear()


class AdaptiveEMA:
    """
    y[n] = α·x[n] + (1−α)·y[n−1]

    α thích nghi: đang vặn khoá (pitch nhảy nhiều) -> α lớn, bám nhanh;
    đã ổn định -> α nhỏ, kim đứng yên.

    Ngưỡng nhảy tính bằng CENT chứ không bằng Hz — bản cũ dùng jump_hz=1.0, mà
    1 Hz ở E2 (82 Hz) là 21 cent còn ở E5 (659 Hz) chỉ là 2.6 cent, tức cùng một
    ngưỡng có nghĩa khác hẳn nhau ở hai đầu dải.
    """
    def __init__(self, a_slow=0.10, a_fast=0.9, jump_cents=25.0):
        self.a_slow, self.a_fast, self.jump = a_slow, a_fast, jump_cents
        self.y = None
    def push(self, x):
        if self.y is None:
            self.y = x
            return x
        a = self.a_fast if abs(cents(x, self.y)) > self.jump else self.a_slow
        self.y = a * x + (1 - a) * self.y
        return self.y
    def reset(self):
        self.y = None

class NoteTracker:
    """
    Toàn bộ state của MỘT NỐT: median, EMA, và bộ đếm frame hụt.

    Gộp ba thứ này vào một lớp vì chúng có CÙNG vòng đời — sinh ra khi bắt đầu
    theo dõi một nốt, chết đi khi người dùng thật sự ngừng chơi. Trước đây chúng
    nằm rải trong Tuner cạnh state của luồng audio, và chính sự lẫn lộn đó khiến
    reset() xoá nhầm cả hai.

    HAI CÁCH GỌI, TƯƠNG ỨNG HAI LOẠI FRAME:
        push(f0)  frame dùng được  -> làm mượt và trả về f0 đã mượt
        miss()    frame không dùng được (im lặng, confidence thấp, subharmonic)

    miss() KHÔNG quên nốt ngay. Nốt đang tắt dần nhấp nháy quanh ngưỡng gate, và
    quên ngay ở frame hụt đầu tiên khiến bộ đệm median không bao giờ đầy nổi —
    median=15 khi đó hành xử y như median=1. Đo được: quên ngay -> jitter p95 tệ
    nhất 25.3 cent; chờ hold_frames rồi mới quên -> 4.93 cent. Xem Config mục C3.
    """

    def __init__(self, cfg: Config):
        self.median = MedianFilter(cfg.med_size)
        self.ema = AdaptiveEMA(cfg.ema_slow, cfg.ema_fast, cfg.ema_jump_c)
        self.hold_frames = cfg.hold_frames
        self._miss = 0

    def push(self, f0):
        """Frame dùng được. Trả f0 đã làm mượt."""
        self._miss = 0
        return self.ema.push(self.median.push(f0))

    def miss(self):
        """Frame không dùng được. Chỉ quên nốt sau khi mất tín hiệu ĐỦ LÂU."""
        self._miss += 1
        if self._miss > self.hold_frames:
            self.median.reset()
            self.ema.reset()

    def reset(self):
        """Quên hẳn nốt đang theo dõi."""
        self.median.reset()
        self.ema.reset()
        self._miss = 0
