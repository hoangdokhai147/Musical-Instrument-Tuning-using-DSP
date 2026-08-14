"""
tuner — Chromatic instrument tuner. Bản rút gọn của dsp_core.py.

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

CẤU TRÚC GÓI (bước 1 của kế hoạch refactor — tách file thuần tuý, không đổi logic)

    core/config.py     Config — mọi tham số, không có hành vi
    core/filters.py    thiết kế FIR lowpass
    core/yin.py        bốn bước YIN + kiểm chứng subharmonic
    core/detector.py   PitchDetector: frame -> f0    (KHÔNG state)
    core/music.py      f0 <-> MIDI <-> tên nốt <-> cent
    core/tracking.py   median + EMA thích nghi        (state của NỐT)
    engine/engine.py   Tuner: ghép tất cả             (state của LUỒNG)
    analysis/          công cụ dev, không nằm trong đường tín hiệu

Quy tắc phụ thuộc: core/ không được import gì ngoài numpy và core/.
"""

from tuner.core.config import Config
from tuner.engine.engine import Tuner

__all__ = ["Config", "Tuner"]
