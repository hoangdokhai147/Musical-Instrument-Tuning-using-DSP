"""Thiết kế FIR. Thuần: nhận số, trả hệ số."""

import numpy as np


# =============================================================================
# 1. FIR LOWPASS — window method
# =============================================================================

def design_lowpass(fs, fc, numtaps):
    """
    Đáp ứng xung lý tưởng của lowpass cắt tại fc là:
        h_ideal[n] = 2·(fc/fs)·sinc(2·(fc/fs)·(n − M/2))
    (biến đổi Fourier ngược của một xung chữ nhật trong miền tần số).
    Nó dài vô hạn, nên nhân với một cửa sổ hữu hạn để cắt bớt:
        h[n] = h_ideal[n] · w[n]
    Hamming cho suy giảm dải chắn ≈ −53 dB, transition width ≈ 3.3/M.

    Chuẩn hoá: chia cho Σh[n] = H(e^{j0}), ép gain tại DC bằng 1.
    (Với LOWPASS thì DC nằm trong dải thông nên đây là điểm chuẩn hoá đúng.
     Bản bandpass cũ phải chiếu lên cosin ở tần số tâm — phức tạp hơn nhiều.)

    numtaps lẻ -> h đối xứng quanh tâm -> pha tuyến tính -> group delay = (M/2) mẫu
    không đổi theo tần số, tức không làm méo dạng sóng.
    """
    if numtaps % 2 == 0:
        numtaps += 1
    m = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = 2 * (fc / fs) * np.sinc(2 * (fc / fs) * m) * np.hamming(numtaps)
    return h / h.sum()


class StreamingFIR:
    """
    Lọc FIR liên tục qua nhiều block, kết quả BIT-EXACT với lọc cả luồng một lần.

    Đây là state của LUỒNG AUDIO — vòng đời khác hẳn state của nốt (median/EMA).
    Nó chỉ được xoá khi mở/đóng stream hoặc đổi thiết bị vào, KHÔNG phải khi người
    dùng ngừng chơi. Tách thành lớp riêng để hai vòng đời đó không thể bị trộn lại;
    việc trộn chúng chính là nguyên nhân của jitter 25 cent (xem Config mục C3).

    Overlap-save: nối len(h)-1 mẫu đuôi của block trước vào đầu block này, rồi dùng
    mode='valid' để chỉ giữ phần KHÔNG bị zero-pad. Bản đầu tiên dùng mode='same'
    mà không có state, nên mỗi biên block có (len(h)-1)/2 mẫu hỏng — đo được sai số
    đỉnh 39.9% so với lọc liên tục.
    """

    def __init__(self, h):
        self.h = h
        self._tail = np.zeros(len(h) - 1)

    def process(self, x):
        xx = np.concatenate([self._tail, x])
        self._tail = xx[-(len(self.h) - 1):]
        return np.convolve(xx, self.h, mode="valid")

    def reset(self):
        self._tail = np.zeros(len(self.h) - 1)