"""PitchDetector: một frame vào, một ước lượng f0 ra. KHÔNG giữ state."""

import numpy as np

from tuner.core.config import Config
from tuner.core.yin import (cmndf, difference_function, find_period,
                            parabolic_interp, subharmonic_db)


# =============================================================================
# 3. PITCH DETECTOR — ghép 4 bước YIN, trả f0 liên tục
# =============================================================================

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
