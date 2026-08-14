"""
Làm mượt f0. Đây là state của NỐT — vòng đời khác với state của LUỒNG (đuôi FIR).
Trộn hai thứ đó chính là nguyên nhân của jitter 25 cent; xem Config mục C3.
"""

import numpy as np

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


# =============================================================================
# 6. TUNER — ghép toàn bộ một frame
# =============================================================================
