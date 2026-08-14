"""Tuner: ghép một frame. Điều phối thuần — không tự giữ state nào."""

import numpy as np

from tuner.core.config import Config
from tuner.core.detector import PitchDetector
from tuner.core.filters import StreamingFIR, design_lowpass
from tuner.core.music import ChromaticResolver
from tuner.core.tracking import NoteTracker
from tuner.engine.framer import Framer


# =============================================================================
# 6. TUNER — ghép toàn bộ một frame
# =============================================================================

class Tuner:
    """
    HAI LOẠI STATE, HAI VÒNG ĐỜI KHÁC NHAU — mỗi loại giờ có một lớp sở hữu nó:

      StreamingFIR   state của LUỒNG. Đuôi bộ lọc. Chỉ xoá khi mở/đóng stream
                     hoặc đổi thiết bị vào.
      NoteTracker    state của NỐT. median, EMA, bộ đếm frame hụt. Xoá khi người
                     dùng thật sự ngừng chơi — KHÔNG phải khi một frame lẻ tụt
                     dưới gate lúc nốt đang tắt dần.
      PitchDetector  KHÔNG state. Cùng frame vào thì luôn cùng kết quả ra.

    Bản đầu giữ cả hai loại state ngay trong Tuner và có một reset() xoá sạch cả
    hai. Đó không phải lỗi thẩm mỹ: nó là nguyên nhân của jitter 25 cent, vì mỗi
    frame tụt gate lại xoá luôn bộ đệm median. Xem Config mục C3.

    Bản thân Tuner giờ không có thuộc tính state nào — hỏi "cái này reset lúc nào"
    thì câu trả lời luôn nằm ở lớp sở hữu nó, không nằm ở đây.
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg if cfg is not None else Config()
        if not isinstance(self.cfg, Config):
            raise TypeError(f"Cần một INSTANCE của Config, nhận {cfg!r}. "
                            f"Có phải bạn viết Config thay vì Config()?")
        self.det = PitchDetector(self.cfg)
        self.resolver = ChromaticResolver(self.cfg.a4)
        self.fir = StreamingFIR(
            design_lowpass(self.cfg.fs, self.cfg.fc_lp, self.cfg.lp_taps))
        self.framer = Framer(self.cfg.buf_len, self.cfg.hop)
        self.tracker = NoteTracker(self.cfg)

    def reset_note(self):
        """Quên nốt đang theo dõi. Gọi khi im lặng đủ lâu, hoặc khi đổi nhạc cụ."""
        self.tracker.reset()

    def reset_stream(self):
        """Quên cả luồng audio. Gọi khi mở/đóng stream hoặc đổi thiết bị vào."""
        self.tracker.reset()
        self.fir.reset()
        self.framer.reset()

    def _drop(self, state, **extra):
        """Một frame không dùng được."""
        self.tracker.miss()
        return {"state": state, **extra}

    def push(self, chunk):
        """
        Nạp audio thô @fs, chunk dài BẤT KỲ (kể cả đổi giữa các lần gọi).
        Trả list gồm 0, 1 hoặc nhiều kết quả — tuỳ chunk chứa được mấy hop.

        Vì sao trả LIST chứ không phải một kết quả: số kết quả phụ thuộc kích
        thước chunk, thứ mà hệ điều hành quyết định chứ không phải ứng dụng.
        Chunk 256 mẫu sinh 0 kết quả ba lần rồi 1 kết quả; chunk 8192 sinh 8.
        Trả list làm điều đó hiển nhiên, thay vì giấu sau một hàm get_result()
        có thể lặng lẽ trả về dữ liệu cũ.
        """
        # Chặn TRƯỚC cả bộ lọc: NaN/Inf sẽ đầu độc đuôi FIR rồi lan sang mọi frame
        # sau đó. Đây là lỗi KỸ THUẬT (buffer hỏng), không phải "audio hợp lệ mà
        # không có cao độ" — nên nó không đụng tới tracker.
        if not np.all(np.isfinite(chunk)):
            return [{"state": "BAD_INPUT"}]

        # LỌC TRƯỚC, CẮT FRAME SAU. Bộ lọc phải thấy luồng đúng một lần, liên tục;
        # nếu cắt frame trước thì nó thấy dữ liệu chồng lấn và state hỏng. Xem
        # framer.py để biết chuyện đó đã gây ra sai số 122.5% biên độ thế nào.
        y = self.fir.process(np.asarray(chunk, dtype=np.float64))
        return [self._process_frame(f) for f in self.framer.push(y)]

    def _process_frame(self, x):
        """Một frame đã lọc, đúng cfg.buf_len mẫu."""
        if float(np.sqrt(np.mean(x ** 2))) < self.cfg.rms_gate:
            return self._drop("SILENT")

        res = self.det.detect(x)
        if res.confidence < self.cfg.min_conf:
            return self._drop("UNRELIABLE", confidence=res.confidence)
        if res.sub_db < -self.cfg.sub_margin:
            # f0 tìm được không có năng lượng trong phổ -> là bội chu kỳ, không phải nốt
            return self._drop("UNRELIABLE", confidence=res.confidence, subharmonic=True)

        f0 = self.tracker.push(res.f0)          # làm mượt trên f0, TRƯỚC khi map
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
