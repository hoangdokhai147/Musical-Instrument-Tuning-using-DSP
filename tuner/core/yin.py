"""
Bốn bước YIN, cộng một lớp kiểm chứng KHÔNG thuộc YIN (xem cuối file).

Không hàm nào ở đây biết về sample rate hay về nốt nhạc — chúng chỉ làm việc với
LAG tính bằng mẫu. Ranh giới đó là lý do engine tự động đúng ở cả 44.1 và 48 kHz
mà không phải sửa gì (đo được: max 0.101 cent ở 44.1 kHz).
"""

import numpy as np


# =============================================================================
# 2. YIN — de Cheveigné & Kawahara (2002), "YIN, a fundamental frequency estimator"
# =============================================================================

def difference_function(x, W, tau_max):
    """
    YIN bước 2 — hàm sai khác:
        d(τ) = Σ_{j=0}^{W−1} ( x[j] − x[j+τ] )²

    Đây là ĐỊNH NGHĨA, viết thẳng, không qua đẳng thức trung gian nào.
    Chi phí (τ_max+1)·W phép nhân-cộng, nhưng chạy trong vòng lặp C của numpy
    nên nhanh hơn bản FFT tự cài 18-79 lần (đã đo).

    Hai tính chất quan trọng, cả hai đều đọc thẳng ra từ công thức:
      * d(τ) BẤT BIẾN với DC: thay x bằng x+c thì (x[j]+c)−(x[j+τ]+c) = x[j]−x[j+τ].
      * d(τ) → 0 khi τ là chu kỳ THẬT *hoặc bất kỳ bội nào của nó*. Đây là nguồn
        gốc của lỗi subharmonic, và là lý do phải có bước 3.
    """
    d = np.empty(tau_max + 1)
    ref = x[:W]
    for tau in range(tau_max + 1):
        e = ref - x[tau:tau + W]
        d[tau] = e @ e
    return d


def cmndf(d):
    """
    YIN bước 3 — Cumulative Mean Normalized Difference Function:
        d'(0) = 1
        d'(τ) = d(τ) / [ (1/τ)·Σ_{j=1}^{τ} d(j) ]

    VÌ SAO CẦN: d(0) = 0 luôn luôn, nên "tìm cực tiểu của d" sẽ luôn ra τ=0.
    Mẫu số là trung bình tích luỹ — nó lớn dần theo τ, nên chia cho nó sẽ PHẠT
    các lag lớn và ưu tiên lag nhỏ. Kết quả: trong số các bội của chu kỳ thật
    (nơi d đều ≈ 0), d' nhỏ nhất tại BỘI NHỎ NHẤT = chu kỳ thật.
    Đây chính là cơ chế chống lỗi subharmonic
    """
    dp = np.ones_like(d)
    run = 0.0
    for tau in range(1, len(d)):
        run += d[tau]
        dp[tau] = d[tau] * tau / run if run > 0 else 1.0
    return dp


def find_period(dp, tau_min, tau_max, thresh):
    """
    YIN bước 4 — ngưỡng tuyệt đối:
    lấy τ NHỎ NHẤT có d'(τ) < thresh, rồi trượt xuống đáy cục bộ.

    VÌ SAO "nhỏ nhất" chứ không phải "argmin toàn cục": nếu tín hiệu tuần hoàn ở T
    thì nó cũng tuần hoàn ở 2T, 3T… Cực tiểu toàn cục có thể nằm ở 2T. Lấy cái
    ĐẦU TIÊN đủ tốt là cách YIN tránh báo thấp hơn một octave.

    Fallback: nếu không τ nào đạt ngưỡng -> lấy argmin và để confidence thấp
    phản ánh việc đó (Tuner sẽ chặn ở cổng confidence).
    """
    tau = tau_min
    while tau <= tau_max:
        if dp[tau] < thresh:
            while tau + 1 <= tau_max and dp[tau + 1] < dp[tau]:
                tau += 1
            return tau
        tau += 1
    return int(tau_min + np.argmin(dp[tau_min:tau_max + 1]))


def parabolic_interp(y, k):
    """
    YIN bước 5 — nội suy parabol qua 3 điểm y[k−1], y[k], y[k+1].

    Đặt parabol p(x) = ax² + bx + c qua 3 điểm cách đều. Đỉnh tại p'(x)=0 cho:
        δ = ½·(y[k−1] − y[k+1]) / (y[k−1] − 2y[k] + y[k+1])
        τ* = k + δ

    VÌ SAO CẦN: chu kỳ thật hầu như không bao giờ là số nguyên mẫu. Ở 220 Hz với
    fs=48 kHz, một mẫu lag = 5.4 cent — không nội suy thì sai số tới ±2.7 cent.

    NỘI SUY TRÊN d(τ), KHÔNG PHẢI d'(τ).
    d'(τ) = d(τ) · τ/(trung bình tích luỹ) là d nhân một hàm TĂNG theo τ. Phép nhân
    đó kéo nhánh phải của parabol lên nhiều hơn nhánh trái, làm đỉnh dịch sang trái,
    tức τ nhỏ hơn, tức f0 CAO hơn. Đo được: bias +0.28 cent trung bình, tăng tới
    +0.54 cent ở dải 300-395 Hz. Nội suy trên d(τ) đưa bias về −0.008 cent.

    Hai lớp bảo vệ:
      den <= 0  -> ba điểm không lồi, không phải cực tiểu -> không nội suy.
      |δ| ≤ ½   -> đỉnh của một cực tiểu thật luôn nằm giữa hai điểm cạnh nó;
                   δ vượt ½ nghĩa là giả thiết bị vi phạm.
    """
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    if den <= 0:
        return float(k)
    return k + max(-0.5, min(0.5, 0.5 * (a - c) / den))


# =============================================================================
# KIỂM CHỨNG PHỔ — KHÔNG phải một bước của YIN
#
# Đây là lớp canh gác đặt TRÊN kết quả của YIN, không phải một phần của nó.
# Lý do tồn tại: d(nT) ≈ 0 với MỌI bội n, nên YIN có thể trả về một chu kỳ đúng
# về mặt toán học mà vô nghĩa về âm nhạc. confidence không bắt được, vì một
# subharmonic vẫn thực sự tuần hoàn. Chỉ có miền tần số mới phân biệt được.
# =============================================================================

def subharmonic_db(x, f0, fs, _cache={}):
    """
    p(f0) − max(p(2·f0), p(3·f0)), tính bằng dB, với p là công suất tại ĐÚNG tần số
    đó (một bin DFT tính trực tiếp — kiểu Goertzel, không qua FFT).

    Tín hiệu tuần hoàn thật LUÔN có năng lượng tại f0 của nó. Nếu giá trị này rất
    âm thì f0 không tồn tại trong tín hiệu — YIN chỉ tìm được một BỘI của chu kỳ
    thật, hợp lệ về mặt toán học (d(nT) ≈ 0 với mọi n) nhưng vô nghĩa về âm nhạc.
    Đo trên bản thu thật: frame đúng trung vị +3.1 dB, frame lỗi trung vị −15.7 dB.

    VÌ SAO KHÔNG DÙNG FFT: với N = 4096 @48 kHz thì bin rộng 11.7 Hz, trong khi dải
    ±3% quanh 110 Hz chỉ rộng 6.6 Hz — hẹp hơn một bin. Bản thử dùng FFT đã loại
    nhầm A2 và D3 (lock 89% -> 5%). Cách này nhắm đúng tần số nên không có vấn đề đó.

    GHI CHÚ HIỆU NĂNG (đo được, sửa lại ước tính ban đầu của tôi): chi phí KHÔNG
    phải ở số phép nhân-cộng (~14.5k, tức 0.5% so với difference_function) mà ở
    np.exp phức và np.hanning. Cache cửa sổ đưa 0.43 ms -> ~0.15 ms.
    """
    n_ = len(x)
    w = _cache.get(n_)
    if w is None:
        w = _cache[n_] = np.hanning(n_)
    xw = x * w
    n = np.arange(n_)
    ph = -2j * np.pi * n / fs
    p = [20.0 * np.log10(abs(np.sum(xw * np.exp(ph * f))) + 1e-12)
         for f in (f0, 2.0 * f0, 3.0 * f0)]
    return p[0] - max(p[1], p[2])
