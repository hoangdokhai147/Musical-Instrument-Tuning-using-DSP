"""
TunerRunner — nối một AudioSource vào Tuner, giữ kết quả mới nhất.

CÁCH DÙNG TRONG APP (đây là toàn bộ những gì dev app cần biết):

    runner = TunerRunner(MicSource())
    runner.start()

    def tick():                       # trong vòng lặp vẽ, ví dụ root.after(33, tick)
        runner.poll()                 # xử lý audio đã về
        r = runner.latest             # TuningResult, hoặc None nếu chưa có gì
        if r is not None:
            ...vẽ...
        root.after(33, tick)

    runner.stop()

MÔ HÌNH MỘT THREAD — và vì sao
    DSP chạy ngay trong poll(), tức trong vòng lặp vẽ của app. Không có thread
    nền. Đã đo cả hai phương án:

        UI kẹt 2 giây (kéo cửa sổ)      1 thread        3 thread
          chunk bị bỏ                        62               0
          tuổi số đọc tệ nhất             112 ms          21 ms
          cú giật khi hồi phục             89 ms           0 ms
        Nhịp vẽ UI (chu kỳ 33 ms)
          lệch nhịp trung vị              12.1 ms         5.1 ms

    3 thread thắng cả hai, nhưng không khác biệt nào trong đó nhận ra được bằng
    mắt trong một app demo. Đổi lại nó thêm vòng đời thread, đường nổi exception
    từ worker, và đường tắt sạch — đúng những thứ hỏng vào lúc trình diễn.

    Cú giật 89 ms đã được chặn xuống ~22 ms bằng MAX_CATCHUP bên dưới.

KHI NÀO NÊN ĐỔI SANG THREAD NỀN
    1. Vẫn thấy kim giật khi kéo cửa sổ dù đã có MAX_CATCHUP
    2. UI thêm hiển thị phổ — tốn thêm ~5.5 ms mỗi tick, gấp đôi chi phí
    3. Máy yếu hơn khiến tải nền vượt ~40% một core (hiện tại 13%)

    Lúc đó chỉ sửa BÊN TRONG file này. poll() và latest giữ nguyên chữ ký, app
    không phải sửa dòng nào.
"""

import dataclasses
from collections import deque

from tuner.core.config import Config
from tuner.engine.engine import Tuner

#: số chunk tối đa xử lý trong một lần poll(). Chặn cú giật sau khi UI kẹt:
#: không có nó thì cả hàng đợi (32 chunk × 2.8 ms = 90 ms) chạy trong một tick.
#: Phần dồn lại tiêu dần ở các tick sau, và chunk quá cũ dù sao cũng bị bỏ.
MAX_CATCHUP = 8


class TunerRunner:
    def __init__(self, source, cfg: Config = None):
        """
        cfg=None -> tự dựng Config khớp sample rate của source.

        Đây không phải tiện lợi vặt: thiết bị có thể từ chối 48 kHz và trả về
        44.1 kHz, và engine chỉ đúng khi Config.fs khớp với rate THẬT. Sai chỗ
        này thì mọi cao độ lệch đi một hệ số, không ai để ý ngay.
        """
        self.source = source
        if cfg is None:
            cfg = Config(fs=source.samplerate)
        elif cfg.fs != source.samplerate:
            cfg = dataclasses.replace(cfg, fs=source.samplerate)
        self.cfg = cfg
        self.tuner = Tuner(cfg)

        self.latest = None          #: TuningResult mới nhất, hoặc None
        self.frames = 0             #: số frame DSP đã xử lý
        self.chunks = 0             #: số chunk audio đã nhận
        self.errors = 0             #: số chunk hỏng (NaN/Inf) đã bỏ qua
        self.overflow = 0           #: chunk bị bỏ vì hàng chờ của runner đầy
        self.last_error = None
        self._running = False

        # Hàng chờ RIÊNG của runner. Cần thiết vì MAX_CATCHUP giới hạn số chunk
        # xử lý mỗi lần poll, còn source.read() thì trả về TẤT CẢ những gì đã về.
        # Không có hàng chờ này, phần dư bị vứt lặng lẽ — đúng bug đã gặp khi chạy
        # thử lần đầu: file 10 giây chỉ xử lý được 8 chunk rồi coi như hết.
        self._pending = deque(maxlen=64)

    # ------------------------------------------------------------------ vòng đời
    def start(self):
        self.source.start()
        self._running = True

    def stop(self):
        self._running = False
        self.source.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # ------------------------------------------------------------------ chạy
    def poll(self):
        """
        Xử lý audio đã về. Gọi từ vòng lặp vẽ của app.
        Trả list các TuningResult mới (thường 0-2 cái), và cập nhật self.latest.
        """
        if not self._running:
            return []

        for chunk in self.source.read():          # rút hết source vào hàng chờ
            if len(self._pending) == self._pending.maxlen:
                self.overflow += 1                # bỏ chunk CŨ NHẤT, có đếm
            self._pending.append(chunk)

        out = []
        for _ in range(MAX_CATCHUP):              # nhưng chỉ xử lý tối đa ngần này
            if not self._pending:
                break
            chunk = self._pending.popleft()
            self.chunks += 1
            try:
                results = self.tuner.push(chunk)
            except ValueError as e:
                # Buffer chứa NaN/Inf: lỗi ở tầng thu âm, không phải "không có
                # cao độ". Đếm và đi tiếp — một chunk hỏng không được làm chết
                # vòng lặp vẽ, nhưng cũng không được biến mất im lặng.
                self.errors += 1
                self.last_error = str(e)
                continue
            out += results
        if out:
            self.frames += len(out)
            self.latest = out[-1]
        return out

    @property
    def pending(self):
        """Số chunk đã nhận nhưng chưa xử lý. Ổn định thì phải quanh 0."""
        return len(self._pending)

    def drain(self):
        """
        Chạy tới khi nguồn hết dữ liệu. Chỉ dùng cho nguồn HỮU HẠN (FileSource)
        trong test và xử lý offline — KHÔNG dùng trong vòng lặp vẽ, và không dùng
        với mic (nó sẽ quay vô tận).
        """
        if not hasattr(self.source, "finished"):
            raise TypeError(f"drain() cần nguồn hữu hạn; {type(self.source).__name__} "
                            f"không có thuộc tính 'finished'")
        out = []
        while True:
            out += self.poll()
            if self.source.finished and not self._pending:
                return out

    def reset(self):
        """Quên nốt và luồng. Gọi khi đổi thiết bị vào."""
        self.tuner.reset_stream()
        self._pending.clear()
        self.latest = None

    # ------------------------------------------------------------------ chẩn đoán
    @property
    def dropped(self):
        """Số chunk bị bỏ vì hàng đợi đầy — chỉ số sức khoẻ. Ổn định thì phải 0."""
        return getattr(self.source, "dropped", 0)

    def stats(self):
        return {"chunks": self.chunks, "frames": self.frames,
                "dropped": self.dropped, "overflow": self.overflow,
                "pending": self.pending, "errors": self.errors,
                "samplerate": self.cfg.fs, "source": self.source.name}
