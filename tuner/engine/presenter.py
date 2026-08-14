"""
Presenter — state của lớp HIỂN THỊ. Loại state thứ ba, sau luồng và nốt.

Ba việc, cả ba đều là chuyện trải nghiệm chứ không phải DSP:

  1. HYSTERESIS ĐỔI NỐT. f0 nằm đúng ranh giới giữa hai nốt (±50 cent) sẽ làm
     tên nốt nhấp nháy qua lại từng frame. Đây là thứ khiến một tuner "cảm giác
     rẻ tiền", và mọi tuner thương mại đều có Schmitt trigger cho nó.

  2. ACQUIRING -> LOCKED. Vài trăm ms đầu sau khi gảy, dây còn pitch glide (gảy
     mạnh làm dây căng thêm nên đầu nốt bị sharp 5-20 cent rồi trôi về đúng).
     Hiện số trong giai đoạn đó thì được, nhưng cho sáng đèn "đúng cao độ" thì
     không.

  3. GIỮ SỐ ĐỌC 800 ms. Người dùng vặn khoá rồi gảy lại; giữa hai lần gảy tín
     hiệu tắt. Nhấp về rỗng mỗi lần là khó chịu, mà báo LOCKED khi không nghe
     thấy gì là nói dối. HOLDING nói đúng: số này cũ stale_ms ms rồi.

VÌ SAO TÁCH KHỎI core/
  core/ không được biết về mili giây, về "hiển thị", hay về việc đèn nào nên
  sáng. Nó chỉ biết mẫu và lag. Ba luật trên đều có thể đổi vì lý do thẩm mỹ mà
  không một dòng DSP nào phải động tới — đó chính là ranh giới đúng.
"""

from tuner.core.config import Config
from tuner.core.music import ChromaticResolver, cents, f0_to_midi, midi_to_freq
from tuner.engine.result import Direction, Status, TuningResult


class Presenter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.resolver = ChromaticResolver(cfg.a4)
        self._held_midi = None    # nốt đang giữ, để hysteresis bám vào
        self._good = 0            # số frame tốt liên tiếp -> quyết định ACQUIRING/LOCKED
        self._in_tune = False     # trạng thái Schmitt của chỉ báo "đúng cao độ"
        self._last = None         # dict các trường đo được của lần đọc tốt gần nhất
        self._last_t = 0.0

    # -------------------------------------------------------------- nốt
    def _resolve_with_hysteresis(self, f0):
        """
        Giữ nốt đang hiện cho tới khi f0 lệch quá 50 + note_hyst_c cent khỏi nó.

        50 cent là ranh giới thật giữa hai nốt; phần cộng thêm là vùng dính. Với
        note_hyst_c = 15 thì có một dải rộng 30 cent quanh ranh giới mà tên nốt
        không đổi. Đổi dây thì vẫn nhảy ngay, vì hai dây cách nhau 400-500 cent.
        """
        nearest = int(round(f0_to_midi(f0, self.cfg.a4)))
        if self._held_midi is None:
            return nearest
        off = abs(cents(f0, midi_to_freq(self._held_midi, self.cfg.a4)))
        return self._held_midi if off <= 50.0 + self.cfg.note_hyst_c else nearest

    def _direction(self, c):
        """
        Schmitt trigger: vào IN_TUNE ở ±in_tune_c, chỉ ra khi vượt ±in_tune_release_c.
        Không có nó thì kim đứng đúng ranh giới sẽ làm đèn chớp liên tục — lỗi UX
        khác với hysteresis đổi nốt, và cần riêng một cơ chế.
        """
        limit = self.cfg.in_tune_release_c if self._in_tune else self.cfg.in_tune_c
        self._in_tune = abs(c) <= limit
        if self._in_tune:
            return Direction.IN_TUNE
        return Direction.FLAT if c < 0 else Direction.SHARP

    # -------------------------------------------------------------- khung kết quả
    def on_pitch(self, f0, raw_f0, conf, level_dbfs, t_ms):
        """Frame có cao độ tin được."""
        midi = self._resolve_with_hysteresis(f0)
        if midi != self._held_midi:
            self._held_midi = midi
            self._good = 0                     # nốt mới -> phải ổn định lại từ đầu
            self._in_tune = False
        self._good += 1

        target = midi_to_freq(midi, self.cfg.a4)
        c = cents(f0, target)
        fields = dict(
            frequency_hz=f0,
            raw_frequency_hz=raw_f0,
            midi_note=midi,
            note_name=self.resolver.name_of(midi),
            octave=midi // 12 - 1,
            target_hz=target,
            cents=c,
            direction=self._direction(c),
        )
        self._last = fields
        self._last_t = t_ms

        status = (Status.LOCKED if self._good >= self.cfg.acquire_frames
                  else Status.ACQUIRING)
        return TuningResult(status=status, confidence=conf, level_dbfs=level_dbfs,
                            timestamp_ms=t_ms, stale_ms=0.0, **fields)

    def on_drop(self, silent, conf, level_dbfs, t_ms):
        """Frame không dùng được. silent=True nếu dưới cổng RMS."""
        if self._last is not None:
            age = t_ms - self._last_t
            if age <= self.cfg.display_hold_ms:
                # Giữ số đọc, nhưng nói rõ nó đã cũ. KHÔNG reset _good: một khoảng
                # hụt ngắn không nên làm mất trạng thái LOCKED.
                return TuningResult(status=Status.HOLDING, confidence=conf,
                                    level_dbfs=level_dbfs, timestamp_ms=t_ms,
                                    stale_ms=age, **self._last)
            self._last = None                  # hết hạn giữ -> quên hẳn
            self._held_midi = None
            self._good = 0
            self._in_tune = False
        return TuningResult(
            status=Status.SILENT if silent else Status.UNSTABLE,
            confidence=conf, level_dbfs=level_dbfs, timestamp_ms=t_ms)

    def reset(self):
        self._held_midi = None
        self._good = 0
        self._in_tune = False
        self._last = None
        self._last_t = 0.0
