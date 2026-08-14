"""Thiết kế FIR. Thuần: nhận số, trả hệ số."""

import numpy as np


# =============================================================================
# 1. FIR LOWPASS — window method (Proakis Ch 10)
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