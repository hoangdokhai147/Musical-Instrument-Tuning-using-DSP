"""
test_state.py — hai loại state phải có hai vòng đời ĐỘC LẬP.

    python tests/test_state.py

VÌ SAO TEST NÀY TỒN TẠI
  Bản đầu tiên gộp state của luồng (đuôi FIR) và state của nốt (median, EMA) vào
  một reset(). Nghe như lỗi thẩm mỹ, thực ra là nguyên nhân của jitter 25 cent:
  mỗi frame tụt dưới gate lại xoá luôn bộ đệm median, nên median=15 chưa bao giờ
  đầy nổi và hành xử y như median=1.

  Golden vectors bắt được hồi quy về SỐ, nhưng không diễn đạt được tính chất
  "hai thứ này phải tách rời". Nếu ai đó sau này gộp lại mà tình cờ vẫn ra cùng
  số trên bộ thu hiện có, golden sẽ xanh còn test này thì đỏ.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.core.config import Config
from tuner.core.filters import StreamingFIR, design_lowpass
from tuner.core.tracking import NoteTracker
from tuner.engine.framer import Framer
from tuner.engine.engine import Tuner


def _tone(f0, n, fs=48000.0, amp=0.3):
    return amp * np.sin(2 * np.pi * f0 * np.arange(n) / fs)


# --- StreamingFIR: lọc theo block phải BẰNG lọc một lần ------------------------

def test_streaming_fir_matches_continuous():
    """Đây là lý do StreamingFIR tồn tại. Bản dùng mode='same' không state sai
    tới 39.9% biên độ ở mỗi biên block."""
    cfg = Config()
    h = design_lowpass(cfg.fs, cfg.fc_lp, cfg.lp_taps)
    x = _tone(220.0, 30000)

    whole = np.convolve(np.concatenate([np.zeros(len(h) - 1), x]), h, mode="valid")

    fir = StreamingFIR(h)
    for block in (4096, 1024, 777):          # kể cả kích thước không tròn
        fir.reset()
        out = np.concatenate([fir.process(x[i:i + block])
                              for i in range(0, len(x), block)])
        m = min(len(whole), len(out))
        err = float(np.max(np.abs(whole[:m] - out[:m])))
        assert err < 1e-12, f"block={block}: sai số {err:.3e}, cần bit-exact"


def test_streaming_fir_reset_clears_tail():
    cfg = Config()
    fir = StreamingFIR(design_lowpass(cfg.fs, cfg.fc_lp, cfg.lp_taps))
    fir.process(_tone(220.0, 5000))
    assert np.any(fir._tail != 0)
    fir.reset()
    assert np.all(fir._tail == 0)


# --- NoteTracker: miss() không được quên NGAY ----------------------------------

def test_tracker_survives_short_dropout():
    """Cốt lõi của fix jitter: vài frame hụt KHÔNG được xoá bộ đệm median."""
    cfg = Config()
    tr = NoteTracker(cfg)
    for _ in range(cfg.med_size):
        tr.push(110.0)
    assert len(tr.median.buf) == cfg.med_size

    for _ in range(cfg.hold_frames):         # hụt nhưng chưa quá ngưỡng
        tr.miss()
    assert len(tr.median.buf) == cfg.med_size, "quên nốt quá sớm — đây chính là bug cũ"
    assert tr.ema.y is not None


def test_tracker_forgets_after_hold():
    cfg = Config()
    tr = NoteTracker(cfg)
    for _ in range(cfg.med_size):
        tr.push(110.0)
    for _ in range(cfg.hold_frames + 1):     # vượt ngưỡng
        tr.miss()
    assert len(tr.median.buf) == 0
    assert tr.ema.y is None


def test_tracker_push_clears_miss_counter():
    cfg = Config()
    tr = NoteTracker(cfg)
    tr.push(110.0)
    for _ in range(cfg.hold_frames):
        tr.miss()
    tr.push(110.0)                            # một frame tốt xoá bộ đếm
    for _ in range(cfg.hold_frames):
        tr.miss()
    assert len(tr.median.buf) > 0, "bộ đếm hụt không được reset khi có frame tốt"


def test_hold_zero_forgets_immediately():
    """hold_frames=0 tái hiện hành vi cũ — phải quên ngay ở frame hụt đầu tiên."""
    import dataclasses
    tr = NoteTracker(dataclasses.replace(Config(), hold_frames=0))
    tr.push(110.0)
    tr.miss()
    assert len(tr.median.buf) == 0 and tr.ema.y is None


# --- Tuner: hai reset phải độc lập ---------------------------------------------

def test_reset_note_leaves_filter_tail_alone():
    """Điều mà bước 3 khẳng định. Trước đây reset() xoá cả hai."""
    tu = Tuner()
    tu.push(_tone(110.0, tu.cfg.buf_len))
    tail_before = tu.fir._tail.copy()

    tu.reset_note()
    assert np.array_equal(tu.fir._tail, tail_before), \
        "reset_note() KHÔNG được đụng tới state của luồng audio"
    assert len(tu.tracker.median.buf) == 0 and tu.tracker.ema.y is None


def test_reset_stream_clears_both():
    tu = Tuner()
    tu.push(_tone(110.0, tu.cfg.buf_len))
    tu.reset_stream()
    assert np.all(tu.fir._tail == 0)
    assert len(tu.tracker.median.buf) == 0 and tu.tracker.ema.y is None


def test_tuner_owns_no_state_itself():
    """Tuner chỉ điều phối. Mọi state phải nằm trong lớp sở hữu nó, để câu hỏi
    'cái này reset lúc nào' luôn có một chỗ trả lời duy nhất."""
    tu = Tuner()
    tu.push(_tone(110.0, tu.cfg.buf_len))
    owned = {k for k, v in vars(tu).items()
             if not isinstance(v, (Config, StreamingFIR, NoteTracker, Framer))
             and type(v).__module__.startswith("tuner")}
    stateful = {k for k, v in vars(tu).items() if k.startswith("_")}
    assert not stateful, f"Tuner tự giữ state: {stateful}"


def test_bad_input_raises_and_does_not_touch_state():
    """NaN là lỗi KỸ THUẬT, không phải 'audio hợp lệ mà không có cao độ'.
    Nó phải NÉM LỖI, không đụng tracker, và không đầu độc đuôi FIR."""
    tu = Tuner()
    tu.push(_tone(110.0, tu.cfg.buf_len))
    tail_before = tu.fir._tail.copy()
    miss_before = tu.tracker._miss
    buf_before = list(tu.tracker.median.buf)

    bad = _tone(110.0, tu.cfg.buf_len).copy()
    bad[100] = np.nan
    try:
        tu.push(bad)
    except ValueError as e:
        assert "NaN" in str(e)
    else:
        raise AssertionError("buffer chứa NaN phải ném ValueError")

    assert np.array_equal(tu.fir._tail, tail_before), "NaN đã lọt vào đuôi FIR"
    assert tu.tracker._miss == miss_before
    assert list(tu.tracker.median.buf) == buf_before


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
