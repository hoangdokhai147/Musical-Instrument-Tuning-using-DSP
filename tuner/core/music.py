"""
Domain logic của âm nhạc: f0 <-> MIDI <-> tên nốt <-> cent.

Không có DSP nào ở đây. Toàn bộ file test được mà không cần một mẫu audio nào.
"""

import numpy as np


# =============================================================================
# 4. NỐT & CENT — chromatic, không giới hạn ở dây buông nào
# =============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def f0_to_midi(f0, a4=440.0):
    """m = 69 + 12·log2(f/A4).  69 là số MIDI của A4; 12 nửa cung một quãng tám."""
    return 69.0 + 12.0 * np.log2(f0 / a4)

def midi_to_freq(m, a4=440.0):
    return a4 * 2.0 ** ((m - 69.0) / 12.0)

def midi_to_name(m):
    """MIDI 60 = C4 (middle C), nên octave = m//12 − 1."""
    m = int(m)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def cents(f_measured, f_target):
    """1200·log2(f/f_ref). Một quãng tám = 1200 cent, một nửa cung = 100 cent."""
    return 1200.0 * np.log2(f_measured / f_target)


class ChromaticResolver:
    """
    Ánh xạ f0 -> nốt gần nhất trong 12 nốt/quãng tám. Không từ chối nốt nào.

    Thay cho GuitarResolver cũ: bản cũ chỉ nhận 6 dây buông và có vùng chết rộng
    200 cent giữa E2 và A2 (reject_cents=150 trong khi khoảng cách dây là 500 cent).
    Bản này nhận MỌI nốt — nên dùng được cho phím bấm, capo, và mọi nhạc cụ khác.
    Sai lệch tối đa so với nốt gần nhất luôn ≤ 50 cent, theo định nghĩa.
    """
    def __init__(self, a4=440.0):
        self.a4 = a4

    def name_of(self, midi):
        """Tên nốt KHÔNG kèm octave — UI render hai phần ở hai cỡ chữ."""
        return NOTE_NAMES[int(midi) % 12]

    def resolve(self, f0):
        m = int(round(f0_to_midi(f0, self.a4)))
        f_target = midi_to_freq(m, self.a4)
        return {"name": midi_to_name(m), "midi": m,
                "target": f_target, "cents": cents(f0, f_target)}
