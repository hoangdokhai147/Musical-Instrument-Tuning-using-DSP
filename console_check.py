#!/usr/bin/env python
"""
console_check.py — chạy engine với audio THẬT và in ra terminal.

    python console_check.py                        # microphone
    python console_check.py --file <đường dẫn>.wav # phát lại một bản thu
    python console_check.py --list                 # liệt kê thiết bị vào

Đây là bằng chứng end-to-end cho bước 6, và là ví dụ mẫu cho dev app: toàn bộ
phần tích hợp nằm gọn trong hàm main() dưới đây, khoảng 15 dòng.
"""

import argparse
import sys
import time

from tuner.engine.result import Status
from tuner.io import FileSource, MicSource, TunerRunner

BAR = 41            # bề rộng thanh cent, lẻ để có ô giữa
ANSI = {"LOCKED": "\033[32m", "ACQUIRING": "\033[33m", "HOLDING": "\033[90m",
        "UNSTABLE": "\033[35m", "SILENT": "\033[90m"}
RESET = "\033[0m"


def bar(cents):
    """Thanh -50..+50 cent với con trỏ. Giống thanh trong mockup UI."""
    pos = int(round((max(-50.0, min(50.0, cents)) + 50.0) / 100.0 * (BAR - 1)))
    cells = ["─"] * BAR
    cells[BAR // 2] = "┼"
    cells[pos] = "●"
    return "".join(cells)


def line(r):
    if r is None:
        return "  đang chờ audio..."
    c = ANSI.get(r.status.value, "")
    if r.frequency_hz is None:
        hint = "chơi một nốt" if r.status is Status.SILENT else "tín hiệu không rõ"
        return (f"  {c}{r.status.value:<9}{RESET} {hint:<28}"
                f"{' ':<41}  {r.level_dbfs:6.1f} dBFS")
    stale = f" (cũ {r.stale_ms:.0f} ms)" if r.stale_ms > 0 else ""
    return (f"  {c}{r.status.value:<9}{RESET} "
            f"{r.note_name + str(r.octave):<4} {r.frequency_hz:8.2f} Hz "
            f"→{r.target_hz:8.2f}  {bar(r.cents)} {r.cents:+6.1f}¢ "
            f"{r.direction.value:<7} conf {r.confidence:.2f}{stale}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="phát lại một .wav thay vì dùng mic")
    ap.add_argument("--device", type=int, help="chỉ số thiết bị vào")
    ap.add_argument("--list", action="store_true", help="liệt kê thiết bị rồi thoát")
    ap.add_argument("--loop", action="store_true", help="lặp lại file")
    a = ap.parse_args()

    if a.list:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    # ---- đây là toàn bộ phần tích hợp mà dev app cần đọc ----
    source = (FileSource(a.file, loop=a.loop) if a.file
              else MicSource(device=a.device))
    runner = TunerRunner(source)

    print(f"\n  ♫ Tuner — {source.name} @ {runner.cfg.fs:.0f} Hz"
          f"   (Ctrl-C để dừng)\n")
    try:
        with runner:
            while True:
                runner.poll()
                print("\r" + line(runner.latest).ljust(150), end="", flush=True)
                if getattr(source, "finished", False) and not runner.poll():
                    break
                time.sleep(0.033)                 # ~30 fps, như vòng lặp vẽ của UI
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:                     # quyền micro, thiết bị hỏng...
        print(f"\n\n  ✗ {e}\n")
        return 1
    # ---------------------------------------------------------

    s = runner.stats()
    print(f"\n\n  {s['frames']} frame · {s['chunks']} chunk · "
          f"{s['dropped']} bỏ · {s['errors']} lỗi")
    if s["dropped"]:
        print("  ⚠ có chunk bị bỏ — máy không theo kịp, hoặc UI bị chặn quá lâu")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
