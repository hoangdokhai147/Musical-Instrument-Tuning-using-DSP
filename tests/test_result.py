"""
test_result.py — hợp đồng TuningResult và ba luật của lớp hiển thị.

    python tests/test_result.py

BỘ THU THẬT KHÔNG KIỂM ĐƯỢC HYSTERESIS. Cả 25 bản thu đều là đàn đã lên dây, nên
không nốt nào nằm gần ranh giới ±50 cent — đo trên chúng cho 13 lần đổi nốt khi
tắt hysteresis và 12 khi bật, tức không kết luận được gì. Nhấp nháy tên nốt chỉ
xảy ra khi f0 đậu ĐÚNG ranh giới, nên phải dựng tín hiệu nhắm thẳng vào đó.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.core.config import Config
from tuner.core.music import midi_to_freq
from tuner.engine.engine import Tuner
from tuner.engine.result import SHOWING, Direction, Status, TuningResult

FS = 48000.0


def _wander(f_center, cents_amp, n, rate_hz=3.0, seed=0):
    """Tín hiệu đi qua đi lại quanh f_center với biên độ cents_amp — mô phỏng dây
    thật rung quanh một cao độ, đủ để đi qua ranh giới nốt nhiều lần."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    dev = cents_amp * np.sin(2 * np.pi * rate_hz * t)
    f = f_center * 2 ** (dev / 1200.0)
    ph = 2 * np.pi * np.cumsum(f) / FS
    y = 0.3 * (np.sin(ph) + 0.4 * np.sin(2 * ph))
    return y + 0.002 * rng.standard_normal(n)


def _run(x, cfg=None, chunk=1024):
    tu = Tuner(cfg or Config())
    out = []
    for i in range(0, len(x), chunk):
        out += tu.push(x[i:i + chunk])
    return out


# --- hợp đồng ------------------------------------------------------------------

def test_contract_holds_on_every_frame():
    """Luật duy nhất: trường đo được khác None KHI VÀ CHỈ KHI status ∈ SHOWING."""
    cfg = Config()
    x = np.concatenate([_wander(220.0, 2.0, 40000), np.zeros(60000),
                        _wander(330.0, 2.0, 40000)])
    for r in _run(x, cfg):
        showing = r.status in SHOWING
        for f in ("frequency_hz", "raw_frequency_hz", "midi_note", "note_name",
                  "octave", "target_hz", "cents", "direction"):
            assert (getattr(r, f) is not None) == showing, \
                f"{r.status}: {f} = {getattr(r, f)!r}"
        assert r.confidence is not None and r.timestamp_ms is not None


def test_contract_is_enforced_at_construction():
    try:
        TuningResult(status=Status.SILENT, confidence=0.0, level_dbfs=-60.0,
                     timestamp_ms=0.0, frequency_hz=110.0)
    except ValueError:
        return
    raise AssertionError("TuningResult phải từ chối SILENT kèm frequency_hz")


def test_timestamp_is_monotonic():
    rs = _run(_wander(220.0, 1.0, 60000))
    ts = [r.timestamp_ms for r in rs]
    assert all(b > a for a, b in zip(ts, ts[1:])), "timestamp phải tăng đơn điệu"


def test_note_name_and_octave_are_separate():
    rs = [r for r in _run(_wander(440.0, 1.0, 40000)) if r.status in SHOWING]
    assert rs and rs[-1].note_name == "A" and rs[-1].octave == 4
    assert rs[-1].midi_note == 69


# --- hysteresis đổi nốt ---------------------------------------------------------

def test_note_hysteresis_stops_boundary_flicker():
    """f0 đậu ĐÚNG ranh giới A3/A#3 và dao động ±8 cent qua lại."""
    boundary = midi_to_freq(57, 440.0) * 2 ** (0.5 / 12)     # giữa A3 và A#3
    x = _wander(boundary, 8.0, 200000)

    flips = {}
    for label, hyst in (("off", 0.0), ("on", 15.0)):
        rs = _run(x, dataclasses.replace(Config(), note_hyst_c=hyst))
        notes = [r.midi_note for r in rs if r.midi_note is not None]
        flips[label] = sum(1 for a, b in zip(notes, notes[1:]) if a != b)

    assert flips["off"] >= 5, \
        f"tín hiệu test chưa đủ gây nhấp nháy (chỉ {flips['off']} lần) — test vô nghĩa"
    assert flips["on"] * 3 <= flips["off"], \
        f"hysteresis không giảm đủ: {flips['off']} -> {flips['on']}"


def test_real_note_change_is_not_blocked():
    """Hysteresis chỉ được dính ở ranh giới, không được chặn đổi dây thật."""
    x = np.concatenate([_wander(110.0, 1.0, 60000), _wander(146.83, 1.0, 60000)])
    rs = [r for r in _run(x) if r.midi_note is not None]
    assert rs[0].midi_note == 45 and rs[-1].midi_note == 50, \
        f"A2 -> D3 phải bám được, nhận {rs[0].midi_note} -> {rs[-1].midi_note}"


# --- hysteresis IN_TUNE ---------------------------------------------------------

def test_in_tune_uses_schmitt_trigger():
    """Vào ở ±5, ra ở ±8. Không có nó thì đèn chớp khi kim đậu đúng ranh giới."""
    cfg = Config()
    tu = Tuner(cfg)
    p = tu.presenter
    seq = [(0.0, Direction.IN_TUNE),       # vào
           (6.0, Direction.IN_TUNE),       # 5 < 6 < 8 -> vẫn giữ
           (7.9, Direction.IN_TUNE),
           (8.5, Direction.SHARP),         # vượt ngưỡng ra
           (6.0, Direction.SHARP),         # 6 > 5 -> chưa vào lại được
           (4.0, Direction.IN_TUNE)]       # dưới ngưỡng vào
    for c, want in seq:
        got = p._direction(c)
        assert got == want, f"cent={c}: nhận {got}, cần {want}"


def test_direction_sign():
    p = Tuner().presenter
    assert p._direction(-40.0) == Direction.FLAT     # thấp hơn -> siết dây
    assert p._direction(+40.0) == Direction.SHARP


# --- ACQUIRING -> LOCKED --------------------------------------------------------

def test_acquiring_then_locked():
    cfg = Config()
    rs = [r for r in _run(_wander(220.0, 1.0, 60000), cfg) if r.status in SHOWING]
    first = [r.status for r in rs[:cfg.acquire_frames]]
    assert all(s == Status.ACQUIRING for s in first[:-1]), \
        f"{cfg.acquire_frames} frame đầu phải là ACQUIRING, nhận {first}"
    assert rs[cfg.acquire_frames - 1].status == Status.LOCKED
    assert rs[-1].status == Status.LOCKED


def test_note_change_returns_to_acquiring():
    """Nốt mới thì phải ổn định lại từ đầu — không được LOCKED ngay."""
    x = np.concatenate([_wander(110.0, 1.0, 80000), _wander(146.83, 1.0, 80000)])
    rs = [r for r in _run(x) if r.status in SHOWING]
    for a, b in zip(rs, rs[1:]):
        if b.midi_note != a.midi_note:
            assert b.status == Status.ACQUIRING, \
                f"đổi nốt {a.midi_note}->{b.midi_note} mà vẫn {b.status}"
            return
    raise AssertionError("không thấy lần đổi nốt nào trong tín hiệu test")


# --- giữ số đọc rồi tắt ---------------------------------------------------------

def test_holding_then_silent():
    cfg = Config()
    x = np.concatenate([_wander(220.0, 1.0, 60000), np.zeros(120000)])
    rs = _run(x, cfg)
    hold = [r for r in rs if r.status == Status.HOLDING]
    assert hold, "phải có giai đoạn HOLDING sau khi mất tín hiệu"
    span = max(r.stale_ms for r in hold)
    assert span <= cfg.display_hold_ms + 1e-6, \
        f"giữ {span:.0f} ms, vượt display_hold_ms = {cfg.display_hold_ms}"
    assert span > cfg.display_hold_ms * 0.8, f"giữ quá ngắn: {span:.0f} ms"
    assert rs[-1].status == Status.SILENT, "hết hạn giữ thì phải về SILENT"


def test_holding_keeps_last_reading_and_marks_it_stale():
    cfg = Config()
    x = np.concatenate([_wander(220.0, 1.0, 60000), np.zeros(30000)])
    rs = _run(x, cfg)
    last_live = [r for r in rs if r.status == Status.LOCKED][-1]
    first_hold = [r for r in rs if r.status == Status.HOLDING][0]
    assert first_hold.frequency_hz == last_live.frequency_hz
    assert first_hold.midi_note == last_live.midi_note
    assert first_hold.stale_ms > 0 and last_live.stale_ms == 0.0


def test_hold_disabled_goes_straight_to_silent():
    cfg = dataclasses.replace(Config(), display_hold_ms=0.0)
    x = np.concatenate([_wander(220.0, 1.0, 60000), np.zeros(30000)])
    rs = _run(x, cfg)
    assert not any(r.status == Status.HOLDING for r in rs)


# --- lỗi kỹ thuật ---------------------------------------------------------------

def test_nan_raises_rather_than_becoming_a_status():
    """Ranh giới đã đặt từ đầu: TuningResult mô tả AUDIO; buffer hỏng là lỗi tầng
    dưới. Nhét nó thành status buộc UI phân biệt 'chưa chơi' với 'app hỏng' bằng
    cách so chuỗi."""
    tu = Tuner()
    bad = _wander(220.0, 1.0, 10000).copy()
    bad[7] = np.nan
    try:
        tu.push(bad)
    except ValueError as e:
        assert "NaN" in str(e)
        return
    raise AssertionError("buffer chứa NaN phải ném ValueError")


def test_as_dict_is_json_ready():
    import json
    rs = [r for r in _run(_wander(220.0, 1.0, 40000)) if r.status in SHOWING]
    d = rs[-1].as_dict()
    json.dumps(d)                                   # không được ném lỗi
    assert d["status"] == "LOCKED" and d["direction"] in ("IN_TUNE", "FLAT", "SHARP")


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
