"""
Phổ hiển thị cho GUI. CÔNG CỤ DEV — không nằm trong đường tín hiệu của sản phẩm.

Tách khỏi core/ vì hai lý do: một lỗi ở đây không thể làm sai số đọc cao độ, và
nó chạy được ở tốc độ khung hình chậm hơn nhiều (đo được 5.5 ms, tức 2.2 lần chi
phí của detect()).
"""

import numpy as np


# =============================================================================
# 7. PHỔ CHO GUI — chỗ duy nhất FFT thực sự cần thiết
# =============================================================================
# FFT tự cài (radix-2 DIT) giữ nguyên từ dsp_core.py. Nó KHÔNG còn nằm trong
# đường pitch nữa, nên một lỗi ở đây không thể làm sai số đọc — và ngược lại,
# nó chạy được ở tốc độ khung hình chậm hơn (ví dụ 10 fps) mà không ảnh hưởng gì.
#
# Ở ĐÂY thì cửa sổ Hann là ĐÚNG và cần thiết — ngược hẳn với YIN:
#   * FFT giả định tín hiệu tuần hoàn ngoài khung. Chỗ nối đầu-cuối không khớp
#     tạo bậc nhảy, và bậc nhảy trải năng lượng ra mọi bin (spectral leakage).
#     Hann đưa hai đầu về 0 nên không có bậc nhảy.
#   * YIN thì KHÔNG được cửa sổ hoá: d(τ) so sánh x[j] với x[j+τ]. Nhân w[j] vào
#     khiến hai mẫu được so chịu hai trọng số khác nhau (w[j] ≠ w[j+τ]), tạo sai
#     khác giả TĂNG theo τ — tức thiên vị lag ngắn, tức gây lỗi octave-too-high.
#     Cửa sổ hoá không chỉ thừa với YIN, nó chủ động có hại.

def _bit_reverse(x):
    n = len(x); j = 0
    x = x.astype(np.complex128)
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit; bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    return x

def fft(x):
    """Cooley-Tukey radix-2 DIT lặp. len(x) phải là luỹ thừa 2."""
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError("FFT cần N = luỹ thừa 2 (hãy zero-pad).")
    X = _bit_reverse(x)
    length = 2
    while length <= n:
        w_len = np.exp(-2j * np.pi / length)
        half = length // 2
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = X[start + k]
                v = X[start + k + half] * w
                X[start + k] = u + v
                X[start + k + half] = u - v
                w *= w_len
        length <<= 1
    return X

def hann(N):
    """w[n] = 0.5·(1 − cos(2πn/(N−1))) — bằng 0 ở hai đầu, nên không có bậc nhảy."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / (N - 1)))

def spectrum(x, fs, n_fft=2048):
    """
    Phổ biên độ một phía cho GUI.
    Trả (freqs, mag_db). Bin k ứng với tần số k·fs/N, k = 0..N/2.
    """
    seg = np.zeros(n_fft)
    take = min(len(x), n_fft)
    seg[:take] = x[:take] * hann(take)
    X = fft(seg)[: n_fft // 2 + 1]
    mag = np.abs(X) / (n_fft / 2)                      # chuẩn hoá về biên độ
    return np.arange(n_fft // 2 + 1) * fs / n_fft, 20 * np.log10(mag + 1e-12)
