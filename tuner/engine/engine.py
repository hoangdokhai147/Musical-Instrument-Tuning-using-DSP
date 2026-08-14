"""Tuner: ghép toàn bộ một frame. Nơi duy nhất giữ state của LUỒNG."""

import numpy as np

from tuner.core.config import Config
from tuner.core.detector import PitchDetector
from tuner.core.filters import design_lowpass
from tuner.core.music import ChromaticResolver
from tuner.core.tracking import AdaptiveEMA, MedianFilter


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
