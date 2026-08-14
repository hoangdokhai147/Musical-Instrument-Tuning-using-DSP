"""
test_config.py — Config phải TỪ CHỐI những cấu hình sai mà engine vẫn chạy được.

    python tests/test_config.py

Không dùng pytest: chưa cài, và bộ test này không cần gì hơn assert. Viết dạng
pytest-compatible (hàm test_*) để sau này cài pytest thì chạy được ngay, không sửa.

VÌ SAO TEST NÀY TỒN TẠI
  Mỗi ca dưới đây là một cách config có thể sai mà engine KHÔNG ném lỗi — nó chạy
  bình thường và cho ra số vô nghĩa. Đó là chế độ hỏng tệ nhất với một dụng cụ đo.
  Ví dụ f_max = 30000 với fs = 48000 cho tau_min = 1, find_period dò từ lag 1, và
  kết quả là rác được báo cáo với confidence cao.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tuner.core.config import Config
from tuner.core.detector import PitchDetector
from tuner.engine.engine import Tuner


# --- cấu hình mặc định phải hợp lệ, nếu không thì mọi thứ khác vô nghĩa ---------

def test_defaults_valid():
    c = Config()
    assert c.tau_min == 43 and c.tau_max == 739 and c.buf_len == 4836


def test_derived_consistent():
    """tau/buf phải suy ra từ fs, không phải hằng số chép tay."""
    for fs, f_min, f_max in ((48000.0, 65.0, 1100.0), (44100.0, 65.0, 1100.0),
                             (48000.0, 80.0, 400.0)):
        c = Config(fs=fs, f_min=f_min, f_max=f_max, fc_lp=min(2000.0, fs / 2 - 1))
        assert c.tau_min == int(fs / f_max)
        assert c.buf_len == c.W + c.tau_max + 1
        assert c.tau_min < c.tau_max


# --- những cấu hình PHẢI bị từ chối --------------------------------------------

BAD = [
    ("f_max vượt Nyquist",       dict(f_max=30000.0)),
    ("f_min >= f_max",           dict(f_min=2000.0)),
    ("f_min âm",                 dict(f_min=-10.0)),
    ("fs âm",                    dict(fs=-48000.0)),
    ("W ngắn hơn tau_max",       dict(W=256)),
    ("hop = 0",                  dict(hop=0)),
    ("lp_taps chẵn",             dict(lp_taps=128)),
    ("lp_taps quá nhỏ",          dict(lp_taps=1)),
    ("fc_lp cắt vào dải f0",     dict(fc_lp=900.0)),
    ("fc_lp vượt Nyquist",       dict(fc_lp=30000.0)),
    ("thresh = 0",               dict(thresh=0.0)),
    ("thresh = 1",               dict(thresh=1.0)),
    ("rms_gate >= 1",            dict(rms_gate=1.5)),
    ("min_conf > 1",             dict(min_conf=1.5)),
    ("sub_margin = 0",           dict(sub_margin=0.0)),
    ("hold_frames âm",           dict(hold_frames=-1)),
    ("med_size chẵn",            dict(med_size=10)),
    ("med_size = 0",             dict(med_size=0)),
    ("ema_slow > ema_fast",      dict(ema_slow=0.95)),
    ("ema_slow = 0",             dict(ema_slow=0.0)),
    ("ema_jump_c âm",            dict(ema_jump_c=-5.0)),
    ("in_tune_c = 0",            dict(in_tune_c=0.0)),
    ("a4 = 0",                   dict(a4=0.0)),
]


def test_rejects_bad_configs():
    for label, kw in BAD:
        try:
            Config(**kw)
        except ValueError:
            continue
        raise AssertionError(f"Config(**{kw}) đáng lẽ phải bị từ chối: {label}")


# --- những cấu hình hợp lệ KHÔNG được từ chối ----------------------------------

GOOD = [
    ("44.1 kHz",            dict(fs=44100.0)),
    ("A4 = 442",            dict(a4=442.0)),
    ("W nhỏ, latency thấp", dict(W=2048)),
    ("dải hẹp guitar",      dict(f_min=75.0, f_max=400.0, fc_lp=900.0)),
    ("không làm mượt",      dict(med_size=1, ema_slow=1.0, ema_fast=1.0)),
    ("không giữ trạng thái", dict(hold_frames=0)),
]


def test_accepts_good_configs():
    for label, kw in GOOD:
        try:
            Config(**kw)
        except ValueError as e:
            raise AssertionError(f"Config hợp lệ bị từ chối ({label}): {e}")


# --- hai cạm bẫy của bản class thường -------------------------------------------

def test_instance_is_frozen():
    """Bản cũ: Config.a4 = 442 đổi cho MỌI instance vì đó là biến class."""
    c = Config()
    try:
        c.a4 = 442.0
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Config phải bất biến")
    assert Config().a4 == 440.0


def test_replace_is_local():
    """Cách đúng để có biến thể: dataclasses.replace — cục bộ, không rò rỉ."""
    a = Config()
    b = dataclasses.replace(a, a4=442.0)
    assert a.a4 == 440.0 and b.a4 == 442.0 and Config().a4 == 440.0


def test_class_instead_of_instance_rejected():
    """Truyền Config thay vì Config() từng nổ tận trong phép toán với thông báo
    'unsupported operand type(s) for /: property and float'. Giờ chặn ngay."""
    for ctor in (PitchDetector, Tuner):
        try:
            ctor(Config)
        except TypeError as e:
            assert "INSTANCE" in str(e)
        else:
            raise AssertionError(f"{ctor.__name__}(Config) phải bị từ chối")


def test_tuner_default_and_explicit_config():
    assert Tuner().cfg == Config()
    assert Tuner(Config(a4=442.0)).cfg.a4 == 442.0


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
    print(f"\n  {len(tests)-failed}/{len(tests)} pass"
          f"    ({len(BAD)} config sai bị từ chối, {len(GOOD)} config hợp lệ được nhận)")
    sys.exit(1 if failed else 0)
