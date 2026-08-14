"""
test_io.py — lớp adapter KHÔNG được làm sai lệch kết quả.

    python tests/test_io.py

Câu hỏi trọng tâm: đưa audio qua source + runner có cho ra đúng cái mà gọi
Tuner.push() trực tiếp cho ra không? Nếu không, mọi số đo chất lượng đã công bố
đều không áp dụng cho app thật.

Không cần microphone. FileSource tồn tại chính vì lý do này — test threading và
buffering mà phải cắm mic thật thì không còn là test.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.core.config import Config
from tuner.engine.engine import Tuner
from tuner.io import FileSource, TunerRunner
from tuner.io.runner import MAX_CATCHUP

WAV = Path(__file__).resolve().parent.parent / "recordings/steady/guitar_A2_open.wav"


class FakeSource:
    """Nguồn điều khiển được, để test riêng phần hàng chờ."""
    samplerate = 48000.0
    name = "fake"

    def __init__(self, chunks):
        self.pending = list(chunks)
        self.dropped = 0
        self.finished = False

    def start(self): pass
    def stop(self): self.finished = True

    def read(self):
        out, self.pending = self.pending, []
        if not out:
            self.finished = True
        return out


# --- điều quan trọng nhất --------------------------------------------------------

def test_runner_matches_direct_push():
    """Qua source + runner phải BẰNG gọi Tuner.push() thẳng, tới từng bit."""
    import soundfile as sf
    x, sr = sf.read(WAV, dtype="float64")
    cfg = Config(fs=sr)

    tu = Tuner(cfg)
    direct = []
    for i in range(0, len(x), 1024):
        direct += tu.push(x[i:i + 1024])

    with TunerRunner(FileSource(WAV, realtime=False)) as r:
        via_runner = r.drain()

    assert len(via_runner) == len(direct), \
        f"{len(via_runner)} kết quả qua runner vs {len(direct)} trực tiếp"
    for k, (a, b) in enumerate(zip(via_runner, direct)):
        assert a.status == b.status, f"frame {k}: {a.status} vs {b.status}"
        assert a.frequency_hz == b.frequency_hz, f"frame {k}: f0 khác"
        assert a.midi_note == b.midi_note, f"frame {k}: nốt khác"


def test_no_chunk_is_silently_lost():
    """MAX_CATCHUP giới hạn công mỗi lần poll, KHÔNG được vứt phần dư.
    Đây là bug thật đã gặp lúc chạy thử: file 10 giây chỉ xử lý 8 chunk."""
    src = FileSource(WAV, realtime=False)
    n_expect = -(-len(src._x) // src.blocksize)      # kể cả chunk đuôi ngắn
    with TunerRunner(src) as r:
        r.drain()
    assert r.chunks == n_expect, f"nhận {r.chunks} chunk, đáng lẽ {n_expect}"
    assert r.overflow == 0 and r.pending == 0


# --- hàng chờ ---------------------------------------------------------------------

def test_poll_is_bounded():
    """Một lần poll không được làm quá MAX_CATCHUP chunk — đây là thứ chặn cú
    giật UI sau khi cửa sổ bị kéo."""
    cfg = Config()
    chunks = [np.zeros(1024) for _ in range(50)]
    r = TunerRunner(FakeSource(chunks), cfg)
    r.start()
    r.poll()
    assert r.chunks == MAX_CATCHUP, f"xử lý {r.chunks} chunk trong một poll"
    assert r.pending == 50 - MAX_CATCHUP


def test_overflow_drops_oldest_and_counts():
    cfg = Config()
    src = FakeSource([])
    r = TunerRunner(src, cfg)
    r.start()
    src.pending = [np.full(1024, float(i)) for i in range(200)]
    r.poll()
    assert r.overflow > 0, "hàng chờ tràn mà không đếm"
    # chunk CŨ nhất bị bỏ -> cái còn lại phải là những cái mới
    assert r.pending == 64 - MAX_CATCHUP


def test_poll_before_start_does_nothing():
    r = TunerRunner(FakeSource([np.zeros(1024)]), Config())
    assert r.poll() == [] and r.chunks == 0


# --- lỗi -------------------------------------------------------------------------

def test_bad_chunk_is_counted_not_fatal():
    """Một chunk hỏng không được làm chết vòng lặp vẽ, cũng không được im lặng."""
    cfg = Config()
    good = np.zeros(1024)
    bad = np.zeros(1024); bad[3] = np.nan
    r = TunerRunner(FakeSource([good, bad, good]), cfg)
    r.start()
    r.poll()
    assert r.errors == 1 and r.last_error and "NaN" in r.last_error
    assert r.chunks == 3, "chunk sau chunk hỏng vẫn phải được xử lý"


def test_drain_refuses_infinite_source():
    class Endless(FakeSource):
        def read(self): return [np.zeros(1024)]
    r = TunerRunner(Endless([]), Config())
    del r.source.finished
    r.start()
    try:
        r.drain()
    except TypeError as e:
        assert "hữu hạn" in str(e)
        return
    raise AssertionError("drain() phải từ chối nguồn vô hạn")


# --- sample rate -------------------------------------------------------------------

def test_config_follows_source_samplerate():
    """Thiết bị có thể từ chối 48 kHz. Config phải bám theo rate THẬT, nếu không
    mọi cao độ lệch đi một hệ số mà không ai để ý."""
    src = FileSource(WAV, realtime=False)
    src.samplerate = 44100.0
    r = TunerRunner(src, Config(fs=48000.0))
    assert r.cfg.fs == 44100.0


# --- FileSource --------------------------------------------------------------------

def test_filesource_yields_the_file_in_order():
    src = FileSource(WAV, realtime=False, blocksize=1024)
    got = []
    src.start()
    while not src.finished:
        got += src.read()
    joined = np.concatenate(got)
    assert np.array_equal(joined, src._x[:len(joined)])


def test_filesource_read_is_bounded():
    src = FileSource(WAV, realtime=False, max_per_read=4)
    src.start()
    assert len(src.read()) == 4, "read() không được đổ cả file ra một lúc"


def test_filesource_realtime_paces_itself():
    src = FileSource(WAV, realtime=True, blocksize=1024, max_per_read=99)
    src.start()
    assert src.read() == [], "vừa start thì chưa có mẫu nào 'tới' cả"
    time.sleep(0.10)
    n = len(src.read())
    expect = 0.10 * src.samplerate / 1024                     # ~4.7 chunk
    assert 0.5 * expect <= n <= 2.0 * expect, f"nhả {n} chunk, kỳ vọng ~{expect:.0f}"


def test_filesource_stops_at_end():
    src = FileSource(WAV, realtime=False)
    src.start()
    while not src.finished:
        src.read()
    assert src.read() == []


# --- vòng đời -----------------------------------------------------------------------

def test_context_manager_stops_source():
    src = FileSource(WAV, realtime=False)
    with TunerRunner(src):
        pass
    assert src.finished


def test_reset_clears_pending_and_latest():
    src = FileSource(WAV, realtime=False)
    r = TunerRunner(src)
    r.start()
    r.poll(); r.poll()
    r.reset()
    assert r.pending == 0 and r.latest is None


# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name}\n      {e}")
    print(f"\n  {len(tests)-failed}/{len(tests)} pass")
    sys.exit(1 if failed else 0)
