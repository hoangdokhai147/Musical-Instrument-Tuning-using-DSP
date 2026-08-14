"""
dsp_simple.py — Chromatic instrument tuner. Bản rút gọn của dsp_core.py.

TRIẾT LÝ: mỗi khối phải giải thích được bằng MỘT công thức, và công thức đó phải
kiểm chứng được bằng một dòng test. Không có khối nào tồn tại chỉ vì "để nhanh hơn".

Khác biệt so với dsp_core.py (và LÝ DO):
  1. KHÔNG hạ mẫu.  fs giữ nguyên 48 kHz.
     -> Xoá: anti-alias reasoning, hệ số decim, fs_eff.
     -> Lợi: độ phân giải lag mịn hơn 4 lần (ở 220 Hz: 5.4 cent/mẫu thay vì 21.7),
        nên nội suy sub-sample phải "gánh" ít hơn -> chính xác hơn.
  2. KHÔNG dùng FFT trong đường pitch.  d(τ) tính thẳng theo định nghĩa.
     -> Xoá: FFT tự cài, bit-reversal, zero-padding chống circular correlation,
        đẳng thức d = P(0)+P(k)−2·CC(k), prefix-sum. (~80 dòng + 3 chủ đề lý thuyết)
     -> Đo được: cách trực tiếp NHANH HƠN 18-79 lần, vì FFT tự cài chạy trong vòng lặp
        Python (~10 lệnh/butterfly) còn cách trực tiếp chạy trong vòng lặp C của numpy
        (~1 lệnh/MAC). Lợi thế O(N log N) bị hằng số ~100:1 nuốt trọn.
        FFT vẫn được giữ — nhưng ở đúng chỗ của nó: tính PHỔ HIỂN THỊ (mục 7).
  3. LOWPASS, không phải bandpass.
     -> Bandpass 50-1500 Hz trong bản cũ có |H(0 Hz)| = −1.35 dB, tức biên dưới
        KHÔNG TỒN TẠI: 127 taps @48 kHz cho transition width 1257 Hz, không thể
        hiện thực hoá corner 50 Hz. Nó vốn đã là lowpass, chỉ là không ai gọi đúng tên.
     -> Lowpass = 1 sinc × 1 cửa sổ ÷ tổng. Bandpass = hiệu 2 sinc + chuẩn hoá cosin.
  4. KHÔNG có dc_block IIR.  Chỉ `x − mean(x)`.
     -> d(τ) = Σ(x[j]−x[j+τ])² BẤT BIẾN TUYỆT ĐỐI với DC: (x+c)−(x+c) = x−x.
        Đã kiểm chứng: lệch đúng 0.00e+00 cent khi thêm DC = +0.5.
        DC chỉ ảnh hưởng cổng RMS, nên trừ trung bình là đủ — và không cần state.
  5. KHÔNG có octave_guard.  Ablation 600 ca khó: 0 lỗi khi bật, 0 lỗi khi tắt.
     Nhưng ở lag nguyên nó bắn nhầm 2-12% số pha vì so sánh tỉ số của hai số ~1e-16.
  6. KHÔNG có GuitarResolver.  Chỉ chromatic -> nhận mọi nốt, không riêng 6 dây buông.

Đo được trên bản này (script s2/s3, sweep liên tục 65-1100 Hz bước 0.5-1 Hz):
    sine sạch          max |err| = 0.072 cent      (dsp_core: 0.896 cent, dải hẹp hơn 3×)
    harmonic + nhiễu   max |err| = 0.446 cent      (dsp_core: 1.272 cent -> TRƯỢT ±1 c)
    SNR 20 dB          MAE = 0.155 cent, 0 lỗi octave
    SNR 10 dB          MAE = 0.696 cent, 0 lỗi octave
    buffer             100.8 ms                    (dsp_core: 185.1 ms)
    thời gian xử lý    ~2.9 ms/frame               (dsp_core: 39.2 ms/frame)

Chỉ dùng numpy cho thao tác mảng. Toàn bộ DSP tự cài.
"""

import numpy as np

# =============================================================================
# 0. CẤU HÌNH — 6 tham số, so với 11 của bản cũ
# =============================================================================

class Config:
    fs        = 48000.0   # sample rate (Hz). KHÔNG hạ mẫu -> không có fs_eff.
    f_min     = 65.0      # C2  — dưới E2 của guitar, phủ cả cello/viola
    f_max     = 1100.0    # ~C#6 — phủ trọn cần đàn guitar và dây buông violin (E5=659)
    W         = 4096      # cửa sổ tham chiếu (mẫu). Xem ghi chú (A) bên dưới.
    hop       = 1024      # bước nhảy giữa hai lần phát hiện (21.3 ms @48k)
    fc_lp     = 2000.0    # cắt lowpass (Hz). Xem ghi chú (B).
    lp_taps   = 127       # bậc FIR (lẻ -> đối xứng -> pha tuyến tính)
    thresh    = 0.15      # ngưỡng tuyệt đối CMNDF (YIN dùng 0.10-0.15)
    a4        = 440.0

    # --- các ngưỡng dưới đây hiệu chuẩn từ 25 bản thu guitar thật, xem mục (C) ---
    rms_gate    = 0.005   # -46 dBFS. Nền phòng đo được -58.8 dBFS -> còn 13 dB biên.
    min_conf    = 0.85    # = 1 - thresh. Xem (C1).
    sub_margin  = 18.0    # dB. Chặn subharmonic. Xem (C2).
    hold_frames = 23      # ~0.5 s. Giữ trạng thái qua khoảng mất tín hiệu ngắn. Xem (C3).
    med_size    = 15      # cửa sổ median
    ema_slow    = 0.10    # alpha khi ổn định
    ema_fast    = 0.90    # alpha khi đang vặn khoá
    ema_jump_c  = 25.0    # cent. Vượt ngưỡng này -> chuyển sang ema_fast.
    in_tune_c   = 5.0     # dung sai hiển thị "đúng cao độ"

    # (A) VÌ SAO W = 4096?
    #     Ràng buộc dưới: d(τ) chỉ có nghĩa khi cửa sổ tham chiếu dài hơn chu kỳ dài nhất.
    #     τ_max = fs/f_min = 739 mẫu, nên W = 4096 cho 5.5 chu kỳ của 65 Hz — thoải mái.
    #     Ràng buộc trên: latency. buffer = W + τ_max = 4835 mẫu = 100.7 ms.
    #     Đo được: W=2048 (58 ms) cũng đạt ±1 cent, chỉ kém hơn khi nhiễu nặng
    #     (SNR 10 dB: MAE 1.07 vs 0.70 cent). Giảm xuống 2048 nếu cần latency thấp.
    #
    # (B) VÌ SAO fc = 2000 Hz?
    #     Chỉ cần > f_max = 1100 Hz đủ margin để không suy giảm f0 cao nhất.
    #     Đo được |H(1100 Hz)| = −0.01 dB, |H(3000 Hz)| = −67 dB.
    #     Vì KHÔNG hạ mẫu nên bộ lọc này KHÔNG phải anti-alias — nó chỉ có một việc:
    #     bỏ công suất nhiễu ngoài dải. Yêu cầu vì thế lỏng hơn nhiều so với bản cũ.
    #
    #     Bộ lọc này KHÔNG thay thế được bằng W dài hơn. Đo được ở SNR 10 dB:
    #         W=2048 không lọc -> 200/259 lỗi octave
    #         W=8192 không lọc -> 197/259 lỗi octave   (gấp 4 lần W, gần như vô ích)
    #         W=2048 CÓ lọc    ->   0/259 lỗi octave
    #     Lý do: nhiễu trắng cộng vào d(τ) một lượng ≈ 2σ²W ở MỌI τ > 0. Tăng W làm
    #     tăng cả số hạng tín hiệu lẫn số hạng nhiễu theo cùng tỉ lệ, nên tỉ số không
    #     đổi. Chỉ giới hạn băng thông (giảm σ²) mới cải thiện được.
    #
    # (C) CÁC NGƯỠNG HIỆU CHUẨN TỪ BẢN THU THẬT (25 take, guitar thép, mic MacBook)
    #
    #     C1. min_conf 0.50 -> 0.85.  find_period chấp nhận lag khi d' < thresh = 0.15,
    #         tức conf > 0.85. Nhưng cổng cũ chỉ loại khi conf < 0.50, nên frame ĐÃ
    #         TRƯỢT ngưỡng YIN (rơi vào nhánh fallback argmin) vẫn được báo là hợp lệ.
    #         Đặt bằng 1 - thresh làm hai ngưỡng nhất quán. Đo: octave 3.8% -> 1.8%.
    #
    #     C2. sub_margin: chặn subharmonic bằng kiểm chứng phổ. Ở frame lỗi, engine báo
    #         109.65 Hz trong khi công suất tại đó là -117.5 dB còn tần số thật
    #         (329.6 Hz) là 0 dB — nó báo một chu kỳ KHÔNG CÓ NĂNG LƯỢNG. confidence
    #         không bắt được vì subharmonic vẫn thực sự tuần hoàn. Đo: 3.8% -> 2.5%,
    #         kết hợp với C1 -> 0.7%.  Phân bố p(f0)-max(p(2f0),p(3f0)):
    #         frame đúng trung vị +3.1 dB, frame lỗi trung vị -15.7 dB.
    #
    #     C3. hold_frames — THAY ĐỔI QUAN TRỌNG NHẤT, và không phải chuyện tham số.
    #         Bản cũ gọi reset() ngay khi MỘT frame rơi dưới gate. Nốt đang tắt dần
    #         nhấp nháy quanh ngưỡng -> bộ đệm median bị xoá liên tục, không bao giờ
    #         kịp đầy 15 mẫu. Vì thế tăng med_size hay giảm ema_slow đều gần như VÔ ÍCH:
    #             quét med_size 1->21  : jitter p95 chỉ giảm 24.8 -> 23.3 cent
    #             quét ema_slow .40->.03: jitter p95 chỉ giảm 25.3 -> 18.3 cent
    #         Chỉ cần ngừng reset ngay lập tức:
    #             hold=0 (cũ) -> jitter p95 tệ nhất 25.3 cent
    #             hold=5      -> 12.8 cent
    #             hold=23     -> 12.8 cent, và A2 từ 25.3 xuống 0.6 cent
    #         Kết hợp với rms_gate -46 dBFS: tệ nhất 5.9 cent, năm trong sáu dây < 1.2.
    #         Bám nốt mới KHÔNG chậm đi (đo trên notechange: 21 ms, không đổi) vì
    #         cơ chế ema_jump_c vẫn kích hoạt ema_fast khi cao độ nhảy.

    @property
    def tau_min(self): return int(self.fs / self.f_max)          # 43
    @property
    def tau_max(self): return int(np.ceil(self.fs / self.f_min)) # 739
    @property
    def buf_len(self): return self.W + self.tau_max + 1          # 4836


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


# =============================================================================
# 2. YIN — bốn bước, mỗi bước một công thức
#    de Cheveigné & Kawahara (2002), "YIN, a fundamental frequency estimator"
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
    Đây chính là cơ chế chống lỗi subharmonic, và nó hoạt động tốt tới mức
    một "octave guard" bổ sung là thừa (đã đo trên 600 ca khó).
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

    NỘI SUY TRÊN d(τ), KHÔNG PHẢI d'(τ) — đây là khác biệt then chốt so với bản cũ.
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
# 3. PITCH DETECTOR — ghép 4 bước YIN, trả f0 liên tục
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
    np.exp phức và np.hanning. Cache cửa sổ đưa 0.43 ms -> ~0.15 ms. Bản port sang
    Dart nên dùng đệ quy Goertzel thật (2 phép nhân/mẫu, không cần exp).
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


class PitchResult:
    __slots__ = ("f0", "confidence", "tau", "sub_db")
    def __init__(self, f0, confidence, tau, sub_db):
        self.f0, self.confidence, self.tau, self.sub_db = f0, confidence, tau, sub_db


class PitchDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def detect(self, x):
        """x: mảng đã lọc, độ dài >= cfg.buf_len. Trả f0 (Hz), độ tin cậy, chỉ số subharmonic."""
        cfg = self.cfg
        if len(x) < cfg.buf_len:
            raise ValueError(f"Cần >= {cfg.buf_len} mẫu, có {len(x)}.")

        x = x - x.mean()                       # DC removal — xem ghi chú ở difference_function
        d   = difference_function(x, cfg.W, cfg.tau_max)
        dp  = cmndf(d)
        tau = find_period(dp, cfg.tau_min, cfg.tau_max, cfg.thresh)

        tau_star = parabolic_interp(d, tau)    # VỊ TRÍ: từ d(τ) — không lệch
        conf = float(np.clip(1.0 - dp[tau], 0.0, 1.0))   # ĐỘ TIN: từ d'(τ) — đã chuẩn hoá [0,1]
        f0 = cfg.fs / tau_star

        # conf đo mức TUẦN HOÀN, KHÔNG đo "đúng octave" — một subharmonic vẫn thực sự
        # tuần hoàn nên vẫn có conf cao. Đó là lý do cần thêm kiểm chứng phổ độc lập.
        return PitchResult(f0, conf, tau_star, subharmonic_db(x, f0, cfg.fs))


# =============================================================================
# 4. NỐT & CENT — chromatic, không giới hạn ở dây buông nào
# =============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def f0_to_midi(f0, a4=440.0):
    """m = 69 + 12·log2(f/A4).  69 là số MIDI của A4; 12 nửa cung một quãng tám."""
    return 69.0 + 12.0 * np.log2(f0 / a4)

def midi_to_freq(m, a4=440.0):
    return a4 * 2.0 ** ((m - 69.0) / 12.0)

def midi_to_name(m):
    """MIDI 60 = C4 (middle C), nên octave = m//12 − 1."""
    m = int(m)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def cents(f_measured, f_target):
    """1200·log2(f/f_ref). Một quãng tám = 1200 cent, một nửa cung = 100 cent."""
    return 1200.0 * np.log2(f_measured / f_target)


class ChromaticResolver:
    """
    Ánh xạ f0 -> nốt gần nhất trong 12 nốt/quãng tám. Không từ chối nốt nào.

    Thay cho GuitarResolver cũ: bản cũ chỉ nhận 6 dây buông và có vùng chết rộng
    200 cent giữa E2 và A2 (reject_cents=150 trong khi khoảng cách dây là 500 cent).
    Bản này nhận MỌI nốt — nên dùng được cho phím bấm, capo, và mọi nhạc cụ khác.
    Sai lệch tối đa so với nốt gần nhất luôn ≤ 50 cent, theo định nghĩa.
    """
    def __init__(self, a4=440.0):
        self.a4 = a4

    def resolve(self, f0):
        m = int(round(f0_to_midi(f0, self.a4)))
        f_target = midi_to_freq(m, self.a4)
        return {"name": midi_to_name(m), "midi": m,
                "target": f_target, "cents": cents(f0, f_target)}


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

class Tuner:
    """
    HAI LOẠI STATE, HAI VÒNG ĐỜI KHÁC NHAU — bản cũ trộn chúng vào một reset():

      state của LUỒNG  : _tail (đuôi FIR). Thuộc về luồng audio, chỉ xoá khi
                         mở/đóng stream hoặc đổi thiết bị.
      state của NỐT    : median, ema. Thuộc về nốt đang chơi, xoá khi người dùng
                         thật sự ngừng chơi — KHÔNG phải khi một frame lẻ tụt
                         dưới gate lúc nốt đang tắt dần.

    Trộn hai thứ này chính là nguyên nhân của jitter 25 cent (xem Config mục C3).
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.det = PitchDetector(self.cfg)
        self.resolver = ChromaticResolver(self.cfg.a4)
        self.median = MedianFilter(self.cfg.med_size)
        self.ema = AdaptiveEMA(self.cfg.ema_slow, self.cfg.ema_fast, self.cfg.ema_jump_c)
        self.lp = design_lowpass(self.cfg.fs, self.cfg.fc_lp, self.cfg.lp_taps)
        self._tail = np.zeros(len(self.lp) - 1)   # state LUỒNG
        self._miss = 0                            # số frame liên tiếp không dùng được

    def _filter(self, x):
        """
        Overlap-save: nối 126 mẫu đuôi của block trước vào đầu block này, rồi
        dùng mode='valid' để chỉ giữ phần KHÔNG bị zero-pad.
        Kết quả khớp CHÍNH XÁC với lọc liên tục — bản cũ dùng mode='same' không
        có state nên mỗi biên block có 63 mẫu hỏng (đo được: sai số đỉnh 39.9%).
        """
        xx = np.concatenate([self._tail, x])
        self._tail = xx[-(len(self.lp) - 1):]
        return np.convolve(xx, self.lp, mode="valid")

    def reset_note(self):
        """Quên nốt đang theo dõi. Gọi khi im lặng đủ lâu, hoặc khi đổi nhạc cụ."""
        self.median.reset(); self.ema.reset(); self._miss = 0

    def reset_stream(self):
        """Quên cả luồng audio. Gọi khi mở/đóng stream hoặc đổi thiết bị vào."""
        self.reset_note()
        self._tail = np.zeros(len(self.lp) - 1)

    def _drop(self, state, **extra):
        """Một frame không dùng được. Chỉ quên nốt sau khi mất tín hiệu ĐỦ LÂU."""
        self._miss += 1
        if self._miss > self.cfg.hold_frames:
            self.median.reset(); self.ema.reset()
        return {"state": state, **extra}

    def process(self, raw):
        """raw: buffer thô @fs, đủ dài để sau lọc còn >= cfg.buf_len mẫu."""
        if not np.all(np.isfinite(raw)):
            return {"state": "BAD_INPUT"}

        x = self._filter(np.asarray(raw, dtype=np.float64))

        if float(np.sqrt(np.mean(x ** 2))) < self.cfg.rms_gate:
            return self._drop("SILENT")

        res = self.det.detect(x)
        if res.confidence < self.cfg.min_conf:
            return self._drop("UNRELIABLE", confidence=res.confidence)
        if res.sub_db < -self.cfg.sub_margin:
            # f0 tìm được không có năng lượng trong phổ -> là bội chu kỳ, không phải nốt
            return self._drop("UNRELIABLE", confidence=res.confidence, subharmonic=True)

        self._miss = 0
        f0 = self.ema.push(self.median.push(res.f0))   # làm mượt trên f0, TRƯỚC khi map
        note = self.resolver.resolve(f0)
        c = note["cents"]
        return {
            "state": "DETECTING",
            "f0": f0,
            "raw_f0": res.f0,
            "note": note["name"],
            "midi": note["midi"],
            "target": note["target"],
            "cents": c,
            "status": ("IN_TUNE" if abs(c) <= self.cfg.in_tune_c
                       else ("LOW" if c < 0 else "HIGH")),
            "confidence": res.confidence,
        }


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
