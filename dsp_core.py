"""
dsp_core.py — Core DSP engine cho musical instrument tuner.

Triết lý thiết kế:
  * Detector chỉ ước lượng f0 (Hz) liên tục. "Nốt" sinh ra ở lớp resolver phía sau.
  * FFT nằm TRONG lõi (autocorrelation qua Wiener-Khinchin), không phải phụ kiện.
  * NACF/CMNDF chạy trên tín hiệu KHÔNG cửa sổ hoá; Hann chỉ dùng cho phổ hiển thị.
  * Làm mượt trên f0 TRƯỚC khi map sang nốt.

Chỉ dùng numpy cho thao tác mảng. Toàn bộ phần DSP (FFT, ACF, CMNDF, FIR) tự cài.
Phần này là REFERENCE IMPLEMENTATION (Python) - Dart port dùng cùng công thức + test vectors.
"""

import numpy as np

# =============================================================================
# 0. CẤU HÌNH
# =============================================================================

class Config:
    fs_in        = 48000      # sample rate đầu vào (Hz)
    decim        = 4          # hệ số hạ mẫu -> fs_eff = 12000 Hz
    f_low        = 60.0       # biên dưới bandpass (Hz)
    f_high       = 1500.0     # biên trên bandpass (Hz) - đồng thời là anti-alias
    fir_taps     = 127        # bậc FIR (lẻ -> pha tuyến tính, đối xứng)
    W            = 2048       # cửa sổ tương quan CỐ ĐỊNH (mẫu, tại fs_eff)
    f_min        = 70.0       # f0 thấp nhất quan tâm (Hz)
    f_max        = 400.0      # f0 cao nhất quan tâm (Hz, chế độ guitar)
    rms_gate     = 0.01       # ngưỡng RMS (biên độ chuẩn hoá [-1,1])
    cmndf_thresh = 0.15       # ngưỡng tuyệt đối CMNDF (YIN dùng ~0.1)
    a4           = 440.0

    @property
    def fs_eff(self):
        return self.fs_in // self.decim


# =============================================================================
# 1. FFT TỰ CÀI — radix-2 DIT lặp (Proakis Ch 8)
# =============================================================================

def _bit_reverse(x):
    n = len(x)
    j = 0
    x = x.astype(np.complex128).copy()
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    return x

def fft(x):
    """DFT qua Cooley-Tukey. len(x) phải là luỹ thừa 2."""
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError("FFT cần N = luỹ thừa 2 (hãy zero-pad).")
    X = _bit_reverse(x)
    length = 2
    while length <= n:
        w_len = np.exp(-2j * np.pi / length)   # twiddle factor
        half = length // 2
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = X[start + k]
                v = X[start + k + half] * w
                X[start + k]        = u + v
                X[start + k + half] = u - v
                w *= w_len
        length <<= 1
    return X

def ifft(X):
    """IDFT = (1/N) conj(FFT(conj(X)))."""
    n = len(X)
    return np.conj(fft(np.conj(X))) / n

def _next_pow2(m):
    p = 1
    while p < m:
        p <<= 1
    return p


# =============================================================================
# 2. THIẾT KẾ FIR BANDPASS — window method (Proakis Ch 10)
#    Hệ số này hard-code / export ra JSON để Python & Dart DÙNG CHUNG.
# =============================================================================

def design_fir_bandpass(fs, f_low, f_high, numtaps):
    """
    FIR pha tuyến tính bằng windowed-sinc.
    h[n] = h_ideal[n] * w[n], với h_ideal là hiệu hai low-pass lý tưởng.
    """
    if numtaps % 2 == 0:
        numtaps += 1                       # ép lẻ -> đối xứng quanh tâm
    M = numtaps - 1
    n = np.arange(numtaps)
    mid = M / 2.0
    fc_low  = f_low  / fs                   # tần số cắt chuẩn hoá [0, 0.5]
    fc_high = f_high / fs

    def ideal_lp(fc):
        # đáp ứng xung low-pass lý tưởng: 2*fc*sinc(2*fc*(n-mid))
        m = n - mid
        return 2 * fc * np.sinc(2 * fc * m)  # np.sinc(x)=sin(pi x)/(pi x)

    h = ideal_lp(fc_high) - ideal_lp(fc_low)   # bandpass = LP_high - LP_low
    w = np.hamming(numtaps)                     # cửa sổ (Ch 10)
    h = h * w
    h /= np.sum(h * np.cos(2*np.pi*(f_low+f_high)/2/fs*(n-mid)))  # chuẩn hoá gain tại tâm dải
    return h


# =============================================================================
# 3. TIỀN XỬ LÝ
# =============================================================================

def dc_block(x, prev_x=0.0, prev_y=0.0, R=0.995):
    """
    DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1]
    H(z) = (1 - z^-1) / (1 - R z^-1): zero tại z=1 (triệt DC), pole tại z=R (gần vòng đơn vị).
    """
    y = np.empty_like(x)
    for n in range(len(x)):
        y[n] = x[n] - prev_x + R * prev_y
        prev_x, prev_y = x[n], y[n]
    return y, prev_x, prev_y

def apply_fir(x, h):
    """Convolution tuyến tính y = x * h (giữ 'same' để đồng bộ độ dài)."""
    return np.convolve(x, h, mode="same")

def downsample(x, factor):
    """Hạ mẫu SAU khi đã bandpass (bandpass kiêm anti-alias)."""
    return x[::factor].copy()

def rms(frame):
    return float(np.sqrt(np.mean(frame ** 2)))


# =============================================================================
# 4. AUTOCORRELATION QUA WIENER-KHINCHIN + CMNDF
# =============================================================================

def autocorr_via_fft(buf, W, K_max):
    """
    Cross-correlation CC[k] = sum_{j=0}^{W-1} buf[j]*buf[j+k], k=0..K_max,
    tính qua FFT: CC = IDFT{ conj(A) * B }, có zero-pad để tránh circular corr.
      A = FFT(buf[0:W]),  B = FFT(buf[0:W+K_max])
    """
    N = _next_pow2(W + K_max + 1)
    a = np.zeros(N); a[:W] = buf[:W]
    b = np.zeros(N); b[:W + K_max] = buf[:W + K_max]
    A = fft(a)
    B = fft(b)
    CC = ifft(np.conj(A) * B).real
    return CC[:K_max + 1]

def difference_function(buf, W, K_max):
    """
    d(k) = sum_{j=0}^{W-1} (buf[j]-buf[j+k])^2
         = P(0) + P(k) - 2*CC(k)          (khai triển bình phương)
      P(0): năng lượng cửa sổ tham chiếu (hằng số)
      P(k): năng lượng cửa sổ dịch k (tính bằng cumsum -> O(L))
      CC(k): cross-correlation (tính bằng FFT)
    """
    CC = autocorr_via_fft(buf, W, K_max)
    sq = buf ** 2
    prefix = np.concatenate(([0.0], np.cumsum(sq)))   # prefix[i]=sum_{0}^{i-1}
    P0 = prefix[W] - prefix[0]
    Pk = prefix[np.arange(K_max + 1) + W] - prefix[np.arange(K_max + 1)]
    d = P0 + Pk - 2 * CC
    return np.maximum(d, 0.0)                          # chống sai số âm nhỏ

def cmndf(d):
    """
    Cumulative Mean Normalized Difference Function (YIN bước 3):
      d'(0)=1;  d'(k)= d(k) / [ (1/k) * sum_{j=1}^{k} d(j) ]
    Mẫu số phạt các lag nhỏ -> triệt bớt lỗi chọn T/2, T/3.
    """
    dp = np.ones_like(d)
    run = 0.0
    for k in range(1, len(d)):
        run += d[k]
        dp[k] = d[k] * k / run if run > 0 else 1.0
    return dp


# =============================================================================
# 5. CHỌN CHU KỲ + NỘI SUY PARABOL
# =============================================================================

def parabolic_interp(y, k):
    """
    Nội suy đỉnh/đáy parabol quanh chỉ số k dựa trên 3 điểm y[k-1],y[k],y[k+1].
    Đỉnh của parabol: k* = k + (y[k-1]-y[k+1]) / (2*(y[k-1]-2*y[k]+y[k+1]))
    Trả về (vị trí nội suy, giá trị nội suy).
    """
    if k <= 0 or k >= len(y) - 1:
        return float(k), float(y[k])
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = (a - 2 * b + c)
    if denom == 0:
        return float(k), float(b)
    delta = 0.5 * (a - c) / denom
    k_star = k + delta
    y_star = b - 0.25 * (a - c) * delta
    return float(k_star), float(y_star)

def find_period(dp, k_min, k_max, thresh):
    """
    YIN absolute threshold: lấy k NHỎ NHẤT có d'(k) < thresh và là đáy cục bộ.
    Nếu không có, lấy đáy toàn cục trong [k_min, k_max].
    """
    k = k_min
    while k <= k_max:
        if dp[k] < thresh:
            while k + 1 <= k_max and dp[k + 1] < dp[k]:
                k += 1                              # trượt xuống đáy cục bộ
            return k
        k += 1
    return int(k_min + np.argmin(dp[k_min:k_max + 1]))

def octave_guard(dp, k, k_max, ratio=1.15):
    """
    Bảo vệ chống chọn nhầm T/2, T/3 (octave-too-high) khi f0 thật yếu:
    nếu 2k, 3k vẫn nằm trong dải và d' ở đó KHÔNG tệ hơn nhiều -> ưu tiên chu kỳ dài hơn.
    Đây là lớp phòng thủ phụ; bandpass low-pass mới là tuyến phòng thủ chính.
    """
    best = k
    for m in (2, 3):
        km = k * m
        if km <= k_max and dp[km] < dp[k] * ratio:
            best = km
    return best


# =============================================================================
# 6. PITCH ENGINE — trả về f0 liên tục, không biết "nốt" là gì
# =============================================================================

class PitchResult:
    __slots__ = ("f0", "confidence", "lag")
    def __init__(self, f0, confidence, lag):
        self.f0, self.confidence, self.lag = f0, confidence, lag

class PitchEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def detect(self, buf, lag_range=None):
        """
        buf: mảng đã tiền xử lý (bandpass + downsample), độ dài >= W + K_max.
        lag_range: (f_min, f_max) tuỳ chế độ. None -> dùng dải mặc định trong cfg.
        """
        cfg = self.cfg
        fs = cfg.fs_eff
        f_min, f_max = lag_range or (cfg.f_min, cfg.f_max)
        k_min = max(1, int(np.floor(fs / f_max)))
        k_max = int(np.ceil(fs / f_min))
        need = cfg.W + k_max + 1
        if len(buf) < need:
            raise ValueError(f"Buffer cần >= {need} mẫu, có {len(buf)}.")

        d  = difference_function(buf, cfg.W, k_max)
        dp = cmndf(d)
        k  = find_period(dp, k_min, k_max, cfg.cmndf_thresh)
        k  = octave_guard(dp, k, k_max)
        k_star, dp_star = parabolic_interp(dp, k)   # nội suy trên CMNDF
        f0 = fs / k_star
        conf = float(np.clip(1.0 - dp_star, 0.0, 1.0))
        return PitchResult(f0, conf, k_star)


# =============================================================================
# 7. RESOLVER — biến f0 thành nốt (đây là lớp guitar / chromatic)
# =============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def f0_to_midi(f0, a4=440.0):
    return 69.0 + 12.0 * np.log2(f0 / a4)

def midi_to_name(m_round):
    idx = int(m_round) % 12
    octave = int(m_round) // 12 - 1
    return f"{NOTE_NAMES[idx]}{octave}"

def cents_to_target(f_measured, f_target):
    return 1200.0 * np.log2(f_measured / f_target)

class ChromaticResolver:
    """Snap f0 về nốt gần nhất trong 12 nốt/quãng tám. Không từ chối."""
    def __init__(self, a4=440.0):
        self.a4 = a4
    def resolve(self, f0):
        m = f0_to_midi(f0, self.a4)
        m_round = round(m)
        f_target = self.a4 * 2 ** ((m_round - 69) / 12.0)
        return {
            "name":   midi_to_name(m_round),
            "target": f_target,
            "cents":  cents_to_target(f0, f_target),
            "valid":  True,
        }

class GuitarResolver:
    """Snap f0 về 6 dây. Có VÙNG TỪ CHỐI: lệch > tol thì báo không hợp lệ."""
    STRINGS = {  # tên -> số MIDI
        "E2": 40, "A2": 45, "D3": 50, "G3": 55, "B3": 59, "E4": 64,
    }
    def __init__(self, a4=440.0, reject_cents=150.0):
        self.a4 = a4
        self.reject_cents = reject_cents
        self.freqs = {n: a4 * 2 ** ((m - 69) / 12.0)
                      for n, m in self.STRINGS.items()}

    def lag_range_for(self, string, semitone_tol=3):
        """Dải tìm lag hẹp quanh 1 dây (Manual Mode) -> triệt bẫy octave."""
        f = self.freqs[string]
        return (f * 2 ** (-semitone_tol / 12.0),
                f * 2 ** ( semitone_tol / 12.0))

    def resolve(self, f0, forced_string=None):
        if forced_string:                    # Manual Mode: target cố định
            name = forced_string
        else:                                # Auto Mode: chọn dây gần nhất
            name = min(self.freqs,
                       key=lambda s: abs(cents_to_target(f0, self.freqs[s])))
        f_target = self.freqs[name]
        c = cents_to_target(f0, f_target)
        return {
            "name":   name,
            "target": f_target,
            "cents":  c,
            "valid":  abs(c) <= self.reject_cents,   # vùng từ chối
        }


# =============================================================================
# 8. HẬU XỬ LÝ — median + EMA thích nghi, chạy TRÊN f0 (trước khi map)
# =============================================================================

class MedianFilter:
    def __init__(self, size=5):
        self.size = size
        self.buf = []
    def push(self, v):
        self.buf.append(v)
        if len(self.buf) > self.size:
            self.buf.pop(0)
        return float(np.median(self.buf))
    def reset(self):
        self.buf.clear()

class AdaptiveEMA:
    """
    EMA thích nghi: đang vặn khoá (|Δf| lớn) -> alpha ~1 bám nhanh;
    ổn định -> alpha nhỏ, kim đứng yên.
    """
    def __init__(self, a_slow=0.25, a_fast=0.9, jump_hz=1.0):
        self.a_slow, self.a_fast, self.jump = a_slow, a_fast, jump_hz
        self.y = None
    def push(self, x):
        if self.y is None:
            self.y = x
            return x
        a = self.a_fast if abs(x - self.y) > self.jump else self.a_slow
        self.y = a * x + (1 - a) * self.y
        return self.y
    def reset(self):
        self.y = None


# =============================================================================
# 9. TUNER SERVICE — ghép toàn bộ luồng 1 frame
# =============================================================================

class TunerService:
    def __init__(self, cfg: Config, mode="guitar_auto"):
        self.cfg = cfg
        self.engine = PitchEngine(cfg)
        self.guitar = GuitarResolver(cfg.a4)
        self.chroma = ChromaticResolver(cfg.a4)
        self.median = MedianFilter(5)
        self.ema = AdaptiveEMA()
        self.mode = mode                 # "guitar_auto" | "guitar_manual" | "chromatic"
        self.forced_string = "E2"
        self.warmup_skip = 2             # bỏ frame đầu (transient / pitch glide)
        self._skipped = 0
        self._dc_state = (0.0, 0.0)
        self.fir = design_fir_bandpass(cfg.fs_in, cfg.f_low, cfg.f_high, cfg.fir_taps)

    def preprocess(self, raw):
        x, px, py = dc_block(raw, *self._dc_state)
        self._dc_state = (px, py)
        x = apply_fir(x, self.fir)          # bandpass kiêm anti-alias
        x = downsample(x, self.cfg.decim)   # hạ mẫu SAU bandpass
        return x

    def process_frame(self, raw_buf):
        """
        raw_buf: buffer thô @fs_in, đủ dài để sau downsample >= W + K_max.
        Trả về dict trạng thái cho GUI.
        """
        x = self.preprocess(raw_buf)

        # --- RMS gate ---
        if rms(x) < self.cfg.rms_gate:
            self._skipped = 0
            self.median.reset(); self.ema.reset()
            return {"state": "LOW_SIGNAL"}

        # --- onset skip: bỏ vài frame đầu sau khi vượt gate ---
        if self._skipped < self.warmup_skip:
            self._skipped += 1
            return {"state": "SETTLING"}

        # --- chọn dải lag theo mode ---
        lag_range = None
        if self.mode == "guitar_manual":
            lag_range = self.guitar.lag_range_for(self.forced_string)

        res = self.engine.detect(x, lag_range)

        # --- confidence gate ---
        if res.confidence < 0.5:
            return {"state": "UNRELIABLE", "confidence": res.confidence}

        # --- LÀM MƯỢT TRÊN f0 (trước khi map) ---
        f0_med = self.median.push(res.f0)
        f0_smooth = self.ema.push(f0_med)

        # --- map sang nốt (một lần, cuối cùng) ---
        if self.mode == "chromatic":
            note = self.chroma.resolve(f0_smooth)
        elif self.mode == "guitar_manual":
            note = self.guitar.resolve(f0_smooth, forced_string=self.forced_string)
        else:  # guitar_auto
            note = self.guitar.resolve(f0_smooth)

        if not note["valid"]:
            return {"state": "UNRELIABLE", "f0": f0_smooth}

        c = note["cents"]
        status = "IN_TUNE" if abs(c) <= 5 else ("LOW" if c < 0 else "HIGH")
        return {
            "state": "DETECTING",
            "f0": f0_smooth,
            "note": note["name"],
            "cents": c,
            "status": status,          # LOW=siết dây, HIGH=nới dây
            "confidence": res.confidence,
        }
