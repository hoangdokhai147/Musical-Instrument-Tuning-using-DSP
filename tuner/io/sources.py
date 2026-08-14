"""
Nguồn audio. Hai lớp, một cách dùng — app không cần biết mình đang chạy nguồn nào.

    MicSource()                    # microphone
    FileSource("....wav")          # phát lại file thu sẵn

VÌ SAO CÓ FileSource
  1. Demo lúc bảo vệ không phụ thuộc mic phòng họp và tiếng ồn khán giả — chạy
     lại đúng một bản thu, kết quả tái lập được.
  2. Test tự động. Không có nó thì muốn test TunerRunner phải cắm mic thật, mà
     như vậy thì không còn là test nữa. Đây mới là lý do chính.

QUY TẮC CỦA CALLBACK AUDIO (chỗ dễ sai nhất trong cả file này)
  Callback chạy trên thread thời gian thực của driver, với deadline vài chục ms.
  Trong đó CHỈ được copy dữ liệu. Không cấp phát lớn, không khoá, không ghi log,
  không chạy DSP. Trễ deadline thì driver nhả tiếng rè ra loa.
  Vì thế callback ở đây làm đúng một việc: append vào deque rồi thoát.
"""

import threading
import time
from collections import deque

import numpy as np


class AudioSource:
    """Interface chung. samplerate và name là thuộc tính bắt buộc."""

    samplerate: float = 0.0
    name: str = "?"

    def start(self):
        raise NotImplementedError

    def read(self):
        """Trả list các chunk đã về (mono, float64). Có thể rỗng."""
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class MicSource(AudioSource):
    """
    Microphone qua sounddevice.

    KÍCH THƯỚC HÀNG ĐỢI có giới hạn và bỏ chunk CŨ NHẤT khi đầy. Với tuner thì
    audio cũ vô giá trị — thà mất 200 ms quá khứ còn hơn chặn callback hoặc hiển
    thị cao độ của nửa giây trước.
    """

    def __init__(self, samplerate=48000.0, device=None, blocksize=1024, maxlen=32):
        import sounddevice as sd
        self._sd = sd
        self.device = device
        self.blocksize = blocksize
        self._q = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stream = None
        self.dropped = 0
        self._silent_reads = 0
        self._checked = False

        info = sd.query_devices(device, kind="input")
        self.name = info["name"]
        try:
            sd.check_input_settings(device=device, samplerate=samplerate,
                                    channels=1, dtype="float32")
            self.samplerate = float(samplerate)
        except Exception:
            # Thiết bị từ chối rate mong muốn -> dùng rate của nó. Engine độc lập
            # sample rate (đã kiểm ở 44.1 và 48 kHz), nên Config chỉ cần khớp lại.
            self.samplerate = float(info["default_samplerate"])

    def _callback(self, indata, frames, time_info, status):
        # CHỈ copy. Mọi thứ khác thuộc về thread khác.
        if len(self._q) == self._q.maxlen:
            self.dropped += 1
        self._q.append(indata[:, 0].astype(np.float64, copy=True))

    def start(self):
        self._stream = self._sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=self.blocksize, device=self.device, callback=self._callback)
        self._stream.start()

    def read(self):
        out = []
        while self._q:
            out.append(self._q.popleft())
        self._check_permission(out)
        return out

    def _check_permission(self, chunks):
        """
        macOS cấp quyền micro cho ỨNG DỤNG TERMINAL, không cho Python. Khi chưa
        cấp, sounddevice KHÔNG ném lỗi — nó trả về toàn số 0. Không bắt ở đây thì
        app hiện SILENT vĩnh viễn và không ai hiểu vì sao.
        """
        if self._checked or not chunks:
            return
        if all(not np.any(c) for c in chunks):
            self._silent_reads += len(chunks)
            if self._silent_reads * self.blocksize > self.samplerate:   # ~1 giây
                raise RuntimeError(
                    "Micro trả về toàn số 0 suốt 1 giây đầu.\n"
                    "  Gần như chắc chắn là QUYỀN MICRO chưa được cấp.\n"
                    "  Cài đặt Hệ thống -> Quyền riêng tư & Bảo mật -> Micrô\n"
                    "  -> bật cho Terminal (hoặc iTerm / VS Code)\n"
                    "  -> KHỞI ĐỘNG LẠI terminal rồi chạy lại.")
        else:
            self._checked = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class FileSource(AudioSource):
    """
    Phát lại một file WAV như thể nó là microphone.

    realtime=True   nhả chunk theo đồng hồ thật — dùng để demo
    realtime=False  nhả nhanh nhất có thể — dùng để test (nhanh và tất định)

    Cả hai chế độ đều giới hạn max_per_read chunk mỗi lần gọi. Nếu không, chế độ
    không-realtime sẽ đổ cả file ra một lúc và làm tràn hàng chờ của bên tiêu thụ —
    tức là test một tình huống không bao giờ xảy ra với mic thật.
    """

    def __init__(self, path, blocksize=1024, realtime=True, loop=False,
                 max_per_read=4):
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        self._x = x
        self.samplerate = float(sr)
        self.name = f"file:{path}"
        self.blocksize = blocksize
        self.realtime = realtime
        self.loop = loop
        self.max_per_read = max_per_read
        self.dropped = 0
        self._pos = 0
        self._t0 = None
        self.finished = False

    def start(self):
        self._pos = 0
        self._t0 = time.perf_counter()
        self.finished = False

    def read(self):
        if self.finished:
            return []
        if self.realtime:
            # bao nhiêu mẫu ĐÁNG LẼ đã tới tính tới thời điểm này
            due = int((time.perf_counter() - self._t0) * self.samplerate)
            end = min(due, len(self._x))
        else:
            end = len(self._x)

        out = []
        while self._pos + self.blocksize <= end and len(out) < self.max_per_read:
            out.append(self._x[self._pos:self._pos + self.blocksize])
            self._pos += self.blocksize

        if self._pos + self.blocksize > len(self._x):
            if self.loop:
                self._pos = 0
                self._t0 = time.perf_counter()
            else:
                # Phần đuôi ngắn hơn một chunk cũng phải giao. Bỏ nó đi thì
                # FileSource không còn tương đương với việc nạp thẳng cả file,
                # và test "runner khớp push trực tiếp" lệch đúng một frame.
                if self._pos < len(self._x) and len(out) < self.max_per_read:
                    out.append(self._x[self._pos:])
                    self._pos = len(self._x)
                if self._pos >= len(self._x):
                    self.finished = True
        return out

    def stop(self):
        self.finished = True
