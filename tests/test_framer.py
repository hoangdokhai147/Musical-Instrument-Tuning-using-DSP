"""
test_framer.py — kết quả KHÔNG được phụ thuộc cách hệ điều hành chia chunk.

    python tests/test_framer.py

Đây là tiêu chí nghiệm thu của bước 4. Kích thước chunk do driver audio quyết
định, đổi giữa các thiết bị, và có thể đổi ngay giữa chừng. Nếu f0 đo được phụ
thuộc vào nó thì engine không dùng được với microphone thật.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.core.config import Config
from tuner.engine.engine import Tuner
from tuner.engine.framer import Framer


def _tone(f0, n, fs=48000.0, amp=0.3):
    t = np.arange(n) / fs
    return amp * (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t))


def _feed(x, sizes):
    """Nạp x vào Tuner theo các kích thước chunk cho trước (lặp vòng)."""
    tu = Tuner()
    out, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out += tu.push(x[i:i + n])
        i += n
        k += 1
    return out


# --- Framer thuần --------------------------------------------------------------

def test_frame_count_and_alignment():
    """N mẫu vào -> đúng 1 + (N - frame_len)//hop frame ra."""
    fr = Framer(100, 25)
    total = 0
    for _ in range(10):
        total += len(fr.push(np.zeros(37)))
    n = 370
    assert total == 1 + (n - 100) // 25 == fr.total_out


def test_frames_are_the_right_samples():
    """Frame thứ k phải đúng là x[k*hop : k*hop + frame_len]."""
    fr = Framer(10, 3)
    x = np.arange(40, dtype=np.float64)
    frames = []
    for i in range(0, len(x), 7):            # chunk 7, không chia hết cho hop
        frames += fr.push(x[i:i + 7])
    for k, f in enumerate(frames):
        want = x[k * 3: k * 3 + 10]
        assert np.array_equal(f, want), f"frame {k}: nhận {f}, cần {want}"


def test_chunk_size_does_not_change_frames():
    fr_ref = Framer(64, 16)
    x = np.random.default_rng(0).standard_normal(1000)
    ref = fr_ref.push(x)
    for size in (1, 3, 16, 64, 65, 333, 1000, 5000):
        fr = Framer(64, 16)
        got = []
        for i in range(0, len(x), size):
            got += fr.push(x[i:i + size])
        assert len(got) == len(ref), f"chunk={size}: {len(got)} frame vs {len(ref)}"
        for a, b in zip(got, ref):
            assert np.array_equal(a, b), f"chunk={size}: nội dung frame khác"


def test_empty_and_oversized_chunks():
    fr = Framer(50, 10)
    assert fr.push(np.zeros(0)) == []
    assert len(fr.push(np.zeros(500))) == 1 + (500 - 50) // 10


def test_reset_clears():
    fr = Framer(50, 10)
    fr.push(np.ones(200))
    fr.reset()
    assert fr.total_in == 0 and fr.total_out == 0
    assert np.all(fr._buf == 0)
    assert len(fr.push(np.ones(49))) == 0      # lại phải chờ đủ frame_len


def test_rejects_bad_geometry():
    for frame_len, hop in ((10, 0), (10, 11), (10, -1)):
        try:
            Framer(frame_len, hop)
        except ValueError:
            continue
        raise AssertionError(f"Framer({frame_len}, {hop}) đáng lẽ bị từ chối")


# --- Tuner end-to-end: đây là tiêu chí nghiệm thu của bước 4 --------------------

def test_tuner_invariant_to_chunk_size():
    """Cùng audio, chia chunk khác nhau -> f0 phải GIỐNG HỆT tới từng bit."""
    x = _tone(146.83, 60000)
    ref = _feed(x, [4096])
    for sizes in ([256], [512], [1024], [100], [8000], [4836]):
        got = _feed(x, sizes)
        assert len(got) == len(ref), f"{sizes}: {len(got)} kết quả vs {len(ref)}"
        for k, (a, b) in enumerate(zip(got, ref)):
            assert a["state"] == b["state"], f"{sizes} frame {k}: state khác"
            if a["state"] == "DETECTING":
                assert a["f0"] == b["f0"], \
                    f"{sizes} frame {k}: f0 {a['f0']} vs {b['f0']}"


def test_tuner_invariant_to_varying_chunk_size():
    """Trường hợp thật nhất: driver đổi kích thước chunk giữa chừng."""
    x = _tone(146.83, 60000)
    ref = _feed(x, [4096])
    rng = np.random.default_rng(7)
    for trial in range(5):
        sizes = list(rng.integers(100, 8000, size=25))
        got = _feed(x, sizes)
        assert len(got) == len(ref), f"lần {trial}: {len(got)} vs {len(ref)}"
        for k, (a, b) in enumerate(zip(got, ref)):
            assert a["state"] == b["state"]
            if a["state"] == "DETECTING":
                assert a["f0"] == b["f0"], f"lần {trial} frame {k}"


def test_first_result_needs_a_full_frame():
    """Không được đoán bừa khi chưa đủ dữ liệu."""
    cfg = Config()
    tu = Tuner(cfg)
    x = _tone(110.0, cfg.buf_len)
    assert tu.push(x[:cfg.buf_len - 1]) == []
    assert len(tu.push(x[cfg.buf_len - 1:cfg.buf_len])) == 1


def test_update_rate_matches_hop():
    """Sau frame đầu, mỗi hop mẫu phải cho đúng một kết quả."""
    cfg = Config()
    tu = Tuner(cfg)
    x = _tone(110.0, cfg.buf_len + 10 * cfg.hop)
    n = len(tu.push(x))
    assert n == 11, f"cần 1 + 10 kết quả, nhận {n}"


def test_bad_input_does_not_enter_framer():
    cfg = Config()
    tu = Tuner(cfg)
    tu.push(_tone(110.0, 2000))
    before = tu.framer.total_in
    bad = _tone(110.0, 2000).copy()
    bad[5] = np.inf
    assert tu.push(bad) == [{"state": "BAD_INPUT"}]
    assert tu.framer.total_in == before, "chunk hỏng đã lọt vào framer"


def test_reset_stream_clears_framer():
    cfg = Config()
    tu = Tuner(cfg)
    tu.push(_tone(110.0, cfg.buf_len))
    tu.reset_stream()
    assert tu.framer.total_out == 0
    assert tu.push(_tone(110.0, cfg.buf_len - 1)) == []


# ------------------------------------------------------------------------------

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
