"""
record_session.py — Thu bộ dữ liệu kiểm thử cho tuner, theo Recording Protocol.

CHẠY:
    ~/miniconda3/envs/dsp/bin/python record_session.py

    # xem thiết bị mà không thu:
    ~/miniconda3/envs/dsp/bin/python record_session.py --devices
    # chọn thiết bị / sample rate khác:
    ~/miniconda3/envs/dsp/bin/python record_session.py --device 1 --sr 48000

VÌ SAO DÙNG SCRIPT THAY VÌ AUDACITY
  * Không có xử lý ẩn. sounddevice -> PortAudio -> CoreAudio, không AGC, không
    noise suppression, không normalize. Đây là lý do chính.
  * Tham số cố định và ghi lại được: mọi bản thu cùng device, cùng sample rate,
    cùng gain. Nếu gain đổi giữa các bản thu thì không hiệu chuẩn được rms_gate.
  * Kiểm tra tự động sau mỗi lần thu: phát hiện clipping, phát hiện file toàn số 0
    (lỗi quyền micro trên macOS - xem dưới), phát hiện thu quá nhỏ.
  * Đặt tên file đúng chuẩn để test suite tự nạp về sau.

CẠM BẪY macOS — ĐỌC KỸ
  1. QUYỀN MICRO. macOS cấp quyền cho ỨNG DỤNG TERMINAL (Terminal/iTerm/VS Code),
     không phải cho Python. Nếu chưa cấp, sounddevice KHÔNG báo lỗi - nó trả về
     TOÀN SỐ 0. Bạn có thể thu 40 phút im lặng mà không biết.
     -> Script này chạy self-test 1 giây trước tiên và dừng ngay nếu thấy toàn 0.
  2. MIC MODE. Control Center -> Micrô -> phải để "Chuẩn" (Standard).
     Nếu đang ở "Cách ly giọng nói" (Voice Isolation), macOS sẽ lọc bỏ mọi thứ
     không giống giọng người - tức là lọc bỏ cây đàn của bạn.
  3. Mic MacBook Air chạy native 96 kHz. Thu ở 48 kHz thì CoreAudio resample.
     Chất lượng resampler của CoreAudio là trong suốt với mục đích này.
"""

import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    sys.exit(f"Thiếu thư viện: {e}\nChạy bằng: ~/miniconda3/envs/dsp/bin/python record_session.py")


# =============================================================================
# DANH SÁCH BẢN THU — mỗi mục trả lời một giả định chưa kiểm chứng
# =============================================================================
# (thư mục, tên file, số giây, bắt buộc?, việc cần làm)

TAKES = [
    # ---- Phần 0: hiệu chuẩn. Ground truth TUYỆT ĐỐI duy nhất của cả buổi. ----
    # Các take có TONE_HZ sẽ được script TỰ PHÁT qua loa Mac rồi thu lại cùng lúc
    # (sd.playrec). Không cần website tone generator, không cần thiết bị thứ hai.
    ("calib", "silence_room",      15, True,
     "KHÔNG chơi gì. Ngồi yên. Phòng đúng như lúc bạn sẽ chỉnh đàn."),
    ("calib", "tone_82",           10, True,
     "Script tự phát 82.41 Hz qua loa. Vặn loa ~50-70%. KHÔNG đổi volume giữa các tone."),
    ("calib", "tone_110",          10, True,
     "Script tự phát 110.00 Hz. GIỮ NGUYÊN volume loa."),
    ("calib", "tone_220",          10, True,
     "Script tự phát 220.00 Hz. GIỮ NGUYÊN volume loa."),
    ("calib", "tone_440",          10, True,
     "Script tự phát 440.00 Hz. GIỮ NGUYÊN volume loa."),
    ("calib", "tone_sweep",        30, False,
     "Script tự phát sweep 80 -> 1000 Hz. GIỮ NGUYÊN volume loa."),

    # ---- Phần 1: độ chính xác steady-state ----
    # CHỈNH LẠI DÂY NGAY TRƯỚC MỖI BẢN THU và ghi số cent tuner báo. Buổi thu đầu
    # chỉnh một lần rồi thu suốt 40 phút -> dây trôi tới 27 cent -> mất ground truth.
    # GẢY LẠI MỖI 1.5-2 GIÂY suốt cả 10 giây. Dây trơn tắt rất nhanh.
    ("steady", "guitar_E2_open",   10, True,  "Dây 6 buông (E2). Gảy lại mỗi ~2 s, đủ mạnh."),
    ("steady", "guitar_A2_open",   10, True,  "Dây 5 buông (A2). Gảy lại mỗi ~2 s."),
    ("steady", "guitar_D3_open",   10, True,  "Dây 4 buông (D3). Gảy lại mỗi ~2 s."),
    ("steady", "guitar_G3_open",   10, True,  "Dây 3 buông (G3). Gảy lại mỗi ~1.5 s — dây trơn, tắt nhanh."),
    ("steady", "guitar_B3_open",   10, True,  "Dây 2 buông (B3). Gảy lại mỗi ~1.5 s — tắt NHANH NHẤT."),
    ("steady", "guitar_E4_open",   10, True,  "Dây 1 buông (E4). Gảy lại mỗi ~1.5 s — dây trơn, tắt nhanh."),
    ("steady", "guitar_A2_flat30", 10, True,
     "Vặn dây A2 XUỐNG ~30 cent (theo tuner tham chiếu) rồi thu. Nhớ vặn lại sau."),

    # ---- Phần 2: transient / time-to-lock ----
    ("transient", "guitar_E2_plucks", 25, False,
     "Gảy E2 ĐÚNG 10 LẦN, cách nhau ~2 giây, để ngân tự nhiên giữa các lần."),
    ("transient", "guitar_E2_hard",   15, False,
     "Dây E2, gảy RẤT MẠNH, 5-6 lần."),
    ("transient", "guitar_E2_soft",   15, False,
     "Dây E2, gảy RẤT NHẸ, 5-6 lần."),

    # ---- Phần 3: các ca khó ----
    ("hard", "guitar_harmonic12_E4",  20, False,
     "Bồi âm: chạm nhẹ dây 1 tại NGĂN 12, gảy, nhả ngón. -> 659 Hz. Làm 5-6 lần."),
    ("hard", "guitar_chromatic_highE", 30, False,
     "Dây 1: buông -> ngăn 1 -> 2 -> ... -> ngăn 12. Mỗi nốt gảy 1 lần, giữ ~2 giây."),
    ("hard", "guitar_notechange",     20, False,
     "Gảy lần lượt E2 -> A2 -> D3 -> G3 -> B3 -> E4, mỗi dây ~3 giây, chuyển dứt khoát."),
    ("hard", "guitar_E2_decay",       30, False,
     "Gảy E2 MỘT LẦN DUY NHẤT rồi để yên cho tới khi tắt hẳn. Đừng gảy thêm."),

    # ---- Phần 4: môi trường ----
    ("env", "guitar_E2_noisy", 15, False,
     "Chơi E2 bình thường NHƯNG bật quạt / mở nhạc nhỏ / có tiếng nói gần đó."),
    ("env", "guitar_E2_quiet", 15, False,
     "Gảy E2 RẤT NHẸ, gần như không nghe rõ. 5-6 lần."),

    # ---- Phần 5: các ca PHẢI bị từ chối ----
    ("reject", "guitar_two_strings", 10, False, "Gảy E2 và A2 CÙNG LÚC, vài lần."),
    ("reject", "guitar_chord_Em",    10, False, "Quạt hợp âm Em, 4-5 lần."),
    ("reject", "guitar_percussive",  10, False, "Gõ thùng đàn, palm mute, va chạm dây - âm không cao độ."),
]

SILENT_TAKES = {"silence_room"}          # được phép (và phải) rất nhỏ
QUIET_TAKES  = {"guitar_E2_quiet"}       # cố tình nhỏ, không cảnh báo

# Take nào script tự phát tone: tên -> tần số Hz (None = sweep)
TONE_HZ = {"tone_82": 82.41, "tone_110": 110.0, "tone_220": 220.0,
           "tone_440": 440.0, "tone_sweep": None}
TONE_AMP = 0.35                          # biên độ phát, chừa headroom


ANALYSIS_SKIP_SEC = 0.20     # bỏ qua bao nhiêu giây đầu file khi PHÂN TÍCH


def synth_tone(f, secs, sr):
    """
    Sine thuần, fade in/out 30 ms để loa không bị 'bụp' ở hai đầu.
    f=None -> sweep loga 80 Hz -> 1000 Hz (loga vì cao độ là thang loga).

    CẢNH BÁO CHO KHÂU PHÂN TÍCH — đã đo được, không phải lý thuyết:
      Fade 30 ms = 1440 mẫu, nằm GỌN TRONG frame 4836 mẫu đầu tiên mà engine đọc.
      Fade là một dạng cửa sổ hoá, và cửa sổ hoá có HẠI cho YIN (x[j] và x[j+tau]
      chịu hai trọng số khác nhau). Hại nặng hơn khi tau lớn, tức f0 thấp:

              f0        phân tích từ mẫu 0     bỏ qua fade
          82.41 Hz          +3.034 cent         -0.032 cent
         110.00 Hz          +1.896 cent         -0.051 cent
         440.00 Hz          +0.139 cent         +0.055 cent

      => LUÔN bỏ qua ANALYSIS_SKIP_SEC giây đầu khi phân tích. Quy tắc này cũng
         đúng cho bản thu guitar: mọi bản thu đều có onset transient (loa nảy,
         dây gảy, phòng dội). Không bỏ qua thì sẽ đo transient rồi tưởng là sai
         số thuật toán.
    """
    n = int(secs * sr)
    t = np.arange(n) / sr
    if f is None:
        f0, f1 = 80.0, 1000.0
        # quét loga: pha = 2*pi*f0*T/ln(k) * (k^(t/T) - 1),  k = f1/f0
        T, k = secs, f1 / f0
        phase = 2 * np.pi * f0 * T / np.log(k) * (k ** (t / T) - 1.0)
    else:
        phase = 2 * np.pi * f * t
    x = TONE_AMP * np.sin(phase)
    fade = int(0.03 * sr)
    if n > 2 * fade:
        w = np.hanning(2 * fade)
        x[:fade] *= w[:fade]
        x[-fade:] *= w[fade:]
    return x.astype(np.float32)


def check_fundamental(x, f, sr):
    """
    Loa nhỏ KHÔNG phát được 82 Hz. Nếu fundamental yếu hơn hài âm bậc 2, thứ bạn
    vừa thu là MÉO CỦA LOA chứ không phải tone bạn định phát — và tuner sẽ báo sai
    một cách hoàn toàn chính đáng.

    Đo năng lượng trong dải hẹp quanh f, 2f, 3f và so sánh.
    Trả (dict mức dB, list cảnh báo).
    """
    n = 1 << int(np.floor(np.log2(min(len(x), sr * 4))))   # floor: n phải <= len(x)
    seg = x[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1 / sr)

    def band(fc, halfwidth=0.03):
        lo, hi = fc * (1 - halfwidth), fc * (1 + halfwidth)
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.max(mag[m])) if m.any() else 0.0

    h = {k: band(f * k) for k in (1, 2, 3)}
    ref = max(h.values()) or 1.0
    db = {f"h{k}_db": 20 * np.log10(v / ref) if v > 0 else -99.0 for k, v in h.items()}

    warn = []
    # So với hài MẠNH NHẤT, không chỉ hài bậc 2. Loa nhỏ bị quá tải đối xứng sinh ra
    # méo BẬC LẺ, nên 3f0 thường mạnh hơn 2f0 — đã gặp thật ở tone_82 trên MacBook Air:
    # h1 = -7 dB, h2 = -40 dB, h3 = 0 dB. Phiên bản cũ chỉ so h1 với h2 nên không báo.
    strongest = max(h[2], h[3])
    if h[1] < strongest:
        k = 2 if h[2] >= h[3] else 3
        warn.append(f"LOA KHÔNG PHÁT ĐƯỢC {f:.0f} Hz — hài bậc {k} mạnh hơn fundamental "
                    f"{db[f'h{k}_db'] - db['h1_db']:.0f} dB. Dùng tai nghe đặt sát mic, "
                    f"hoặc bỏ qua tone này.")
    elif db["h2_db"] > -6 or db["h3_db"] > -6:
        warn.append(f"Méo loa đáng kể (2f0 {db['h2_db']:+.0f}, 3f0 {db['h3_db']:+.0f} dB). "
                    f"Kết quả cent ở tone này kém tin cậy.")
    return db, warn


# =============================================================================
# KIỂM TRA CHẤT LƯỢNG SAU MỖI LẦN THU
# =============================================================================

RMS_GATE_DBFS = -40.0        # bằng cfg.rms_gate = 0.01 của dsp_simple.py
MIN_USABLE_PCT = 30.0        # ngưỡng cho các take CẦN tín hiệu liên tục

# Chỉ các take dùng để ĐO CENT mới cần tín hiệu liên tục. Các take còn lại CỐ Ý có
# khoảng lặng — gảy rời từng nhát, decay tắt dần, bồi âm vốn nhỏ, ca phải bị từ chối —
# nên áp quy tắc này cho chúng sẽ báo oan (guitar_E2_plucks chỉ 27% nhưng đạt 98% lock).
SUSTAIN_TAKES = {
    "guitar_E2_open", "guitar_A2_open", "guitar_D3_open",
    "guitar_G3_open", "guitar_B3_open", "guitar_E4_open",
    "guitar_A2_flat30", "guitar_notechange", "guitar_chromatic_highE",
    "guitar_E2_noisy",
}


def usable_fraction(x, sr, gate_dbfs=RMS_GATE_DBFS, frame_ms=50):
    """
    % thời lượng có RMS trên ngưỡng gate của engine.

    ĐÂY MỚI LÀ SỐ ĐO ĐÚNG cho bản thu guitar, không phải peak dBFS.
    Một lần gảy mạnh rồi tắt ngay cho peak đẹp nhưng engine không dùng được gì:
    buổi thu đầu có guitar_B3_open peak -29.5 dBFS (không đủ để cảnh báo) nhưng
    chỉ 1% thời lượng trên gate -> 0 frame nào qua nổi cổng RMS.
    Dây trơn (G3/B3/E4) tắt nhanh hơn dây quấn nhiều, nên phải gảy lại liên tục.
    """
    n = max(1, int(frame_ms/1000*sr))
    if len(x) < n:
        return 0.0
    e = np.array([np.sqrt(np.mean(x[i:i+n]**2)) for i in range(0, len(x)-n, n)])
    thr = 10 ** (gate_dbfs/20)
    return 100.0 * float(np.mean(e > thr))


def analyse(x, name, sr=48000):
    """Trả (dict thống kê, list cảnh báo, có nên thu lại không)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms  = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    dbfs = 20*np.log10(peak) if peak > 0 else -np.inf
    rdb  = 20*np.log10(rms)  if rms  > 0 else -np.inf
    clipped = int(np.sum(np.abs(x) >= 0.999))
    usable = usable_fraction(x, sr)
    stats = {"peak": peak, "peak_dbfs": dbfs, "rms": rms, "rms_dbfs": rdb,
             "clipped_samples": clipped, "usable_pct": usable}

    warn, redo = [], False
    if peak < 1e-6:
        warn.append("TOÀN SỐ 0 — micro không được cấp quyền, hoặc sai thiết bị.")
        redo = True
    elif name in SILENT_TAKES:
        pass                                          # im lặng là đúng chủ đích
    elif name in QUIET_TAKES:
        if peak < 0.002:
            warn.append("Nhỏ tới mức có thể không phải tín hiệu. Kiểm tra lại.")
    elif name in TONE_HZ:
        if dbfs < -35:
            warn.append(f"Đỉnh {dbfs:.1f} dBFS — vặn loa to hơn.")
        elif dbfs > -3:
            warn.append(f"Đỉnh {dbfs:.1f} dBFS — vặn loa nhỏ lại.")
    else:
        if clipped > 0:
            warn.append(f"CLIPPING: {clipped} mẫu chạm trần. Gảy nhẹ hơn / lùi mic.")
            redo = True
        elif dbfs > -3:
            warn.append(f"Đỉnh {dbfs:.1f} dBFS — quá sát trần.")
        # DÙNG ĐƯỢC BAO NHIÊU mới là điều quan trọng, không phải đỉnh cao bao nhiêu
        if name in SUSTAIN_TAKES and usable < MIN_USABLE_PCT:
            warn.append(f"CHỈ {usable:.0f}% THỜI LƯỢNG TRÊN GATE (cần ≥{MIN_USABLE_PCT:.0f}%) — "
                        f"engine sẽ bỏ qua phần còn lại. Gảy LẠI mỗi 1.5-2 giây, "
                        f"mạnh hơn, mic gần hơn.")
            redo = True
    return stats, warn, redo


def write_digital_refs(root, sr):
    """
    Ghi tone THUẦN SỐ ra WAV — không qua loa, không qua mic, không qua không khí.

    Đây là nhóm đối chứng: nếu engine đọc các file này mà sai cent, lỗi nằm ở
    engine hoặc ở khâu đọc WAV. Nếu các file này đúng mà bản thu qua mic sai,
    lỗi nằm ở chuỗi âm học. Tách được hai nguyên nhân bằng 4 file sinh trong 1 giây.
    """
    d = root / "digital"
    if d.exists() and any(d.glob("*.wav")):
        return
    d.mkdir(parents=True, exist_ok=True)
    for f in (82.41, 110.0, 220.0, 440.0, 659.25, 1046.50):
        sf.write(d / f"digital_{f:.2f}Hz.wav", synth_tone(f, 5.0, sr), sr, subtype="PCM_24")
    # thêm một tín hiệu giống guitar: 6 hài âm, biên độ 1/k
    t = np.arange(int(5 * sr)) / sr
    y = sum((1.0 / k) * np.sin(2 * np.pi * 82.41 * k * t) for k in range(1, 7))
    y = (0.35 * y / np.max(np.abs(y))).astype(np.float32)
    sf.write(d / "digital_E2_harmonic.wav", y, sr, subtype="PCM_24")
    print(f"  ✓ Đã sinh 7 file đối chứng thuần số trong {d}/\n")


def selftest(device, sr):
    """Thu 1 giây và xác nhận có tín hiệu thật. Bắt lỗi quyền micro của macOS."""
    print("  Đang thu thử 1 giây... (hãy nói hoặc gõ nhẹ vào bàn)")
    try:
        x = sd.rec(int(sr), samplerate=sr, channels=1, dtype="float32", device=device)
        sd.wait()
    except Exception as e:
        print(f"\n  ✗ Không mở được luồng thu: {type(e).__name__}: {e}")
        return False

    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        print(f"""
  ✗ THU ĐƯỢC TOÀN SỐ 0 (đỉnh = {peak:.2e})

    Gần như chắc chắn là QUYỀN MICRO. macOS cấp quyền cho ứng dụng terminal,
    không phải cho Python.

    Sửa: Cài đặt Hệ thống -> Quyền riêng tư & Bảo mật -> Micrô
         -> bật cho Terminal (hoặc iTerm / Visual Studio Code — app nào bạn đang chạy)
         -> KHỞI ĐỘNG LẠI terminal rồi chạy lại script này.
""")
        return False
    print(f"  ✓ Có tín hiệu. Đỉnh = {20*np.log10(peak):.1f} dBFS\n")
    return True


def countdown(seconds, label):
    for r in range(seconds, 0, -1):
        print(f"\r  ● ĐANG THU {label} — còn {r:2d} s ", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 52 + "\r", end="")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", action="store_true", help="liệt kê thiết bị rồi thoát")
    ap.add_argument("--device", type=int, default=None, help="chỉ số thiết bị vào")
    ap.add_argument("--sr", type=int, default=48000, help="sample rate (mặc định 48000)")
    ap.add_argument("--out", default="recordings", help="thư mục ra")
    ap.add_argument("--only", default=None, help="chỉ thu nhóm này: calib/steady/transient/hard/env/reject")
    ap.add_argument("--redo", default=None,
                    help="xoá rồi thu lại các take này, ngăn cách bằng dấu phẩy. "
                         "Ví dụ: --redo guitar_G3_open,guitar_B3_open,guitar_E4_open")
    args = ap.parse_args()

    if args.devices:
        print(sd.query_devices())
        return

    device = args.device if args.device is not None else sd.default.device[0]
    info = sd.query_devices(device)
    sr = args.sr
    root = Path(args.out)

    print("=" * 74)
    print("  THU DỮ LIỆU KIỂM THỬ TUNER")
    print("=" * 74)
    print(f"  Thiết bị    : [{device}] {info['name']}")
    print(f"  Native rate : {info['default_samplerate']:.0f} Hz"
          + ("   (CoreAudio sẽ resample — không sao)" if info['default_samplerate'] != sr else ""))
    print(f"  Thu ở       : {sr} Hz, mono, float32 -> WAV 24-bit")
    print(f"  Thư mục ra  : {root.resolve()}")
    print()
    print("  TRƯỚC KHI BẮT ĐẦU:")
    print("    1. Control Center -> Micrô -> để 'Chuẩn' (KHÔNG phải Cách ly giọng nói)")
    print("    2. Vặn volume loa ~50-70% và GIỮ NGUYÊN suốt 5 take tone đầu")
    print("       (script tự phát tone qua loa Mac rồi thu lại — không cần website)")
    print("    3. Đặt mic cách đàn ~30 cm và GIỮ NGUYÊN vị trí cả buổi")
    print("    4. Chỉnh đàn bằng tuner có sẵn, chép lại số cent từng dây")
    print("=" * 74)
    print()

    write_digital_refs(root, sr)

    if not selftest(device, sr):
        sys.exit(1)

    takes = [t for t in TAKES if args.only is None or t[0] == args.only]

    if args.redo:
        want = {s.strip() for s in args.redo.split(",") if s.strip()}
        known = {t[1] for t in TAKES}
        for bad in want - known:
            sys.exit(f"Không có take tên '{bad}'. Các take hợp lệ:\n  "
                     + "\n  ".join(sorted(known)))
        for g, n, *_ in TAKES:
            if n in want:
                p = root / g / f"{n}.wav"
                if p.exists():
                    p.unlink(); print(f"  đã xoá {p} — sẽ thu lại")
        takes = [t for t in takes if t[1] in want]
        print()

    done, skipped = [], []

    for i, (group, name, secs, required, what) in enumerate(takes, 1):
        path = root / group / f"{name}.wav"
        tag = "BẮT BUỘC" if required else "tuỳ chọn"

        if path.exists():
            print(f"[{i:2d}/{len(takes)}] {name}  — đã có, bỏ qua "
                  f"(xoá file nếu muốn thu lại)")
            continue

        print("─" * 74)
        print(f"[{i:2d}/{len(takes)}] {group}/{name}.wav   ({secs} s · {tag})")
        print(f"         {what}")
        while True:
            r = input("         [Enter]=thu   s=bỏ qua   q=dừng buổi thu  > ").strip().lower()
            if r == "q":
                print("\nDừng lại. Chạy lại script để tiếp tục từ đây.")
                _write_meta(root, device, info, sr, done, skipped)
                return
            if r == "s":
                skipped.append(name)
                print("         → bỏ qua\n")
                break

            print("         Bắt đầu sau 3...", end="", flush=True); time.sleep(1)
            print(" 2...", end="", flush=True); time.sleep(1)
            print(" 1...", flush=True); time.sleep(1)

            if name in TONE_HZ:
                # Vừa phát vừa thu trên CÙNG một máy. Cùng device nên cùng clock
                # -> không có sai lệch sample rate giữa phát và thu.
                tone = synth_tone(TONE_HZ[name], secs, sr)
                x = sd.playrec(tone.reshape(-1, 1), samplerate=sr, channels=1,
                               dtype="float32", device=(device, sd.default.device[1]))
            else:
                x = sd.rec(int(secs * sr), samplerate=sr, channels=1,
                           dtype="float32", device=device)
            countdown(secs, name)
            sd.wait()
            x = x[:, 0]

            stats, warns, redo = analyse(x, name)
            if name in TONE_HZ and TONE_HZ[name] is not None:
                hdb, hwarn = check_fundamental(x, TONE_HZ[name], sr)
                stats.update(hdb)
                warns += hwarn
            print(f"         đỉnh {stats['peak_dbfs']:6.1f} dBFS   "
                  f"rms {stats['rms_dbfs']:6.1f} dBFS   "
                  f"clip {stats['clipped_samples']}"
                  + (f"   hài: f0 {stats['h1_db']:+.0f} / 2f0 {stats['h2_db']:+.0f} / "
                     f"3f0 {stats['h3_db']:+.0f} dB" if "h1_db" in stats else ""))
            for w in warns:
                print(f"         ⚠  {w}")

            if redo:
                print("         → PHẢI thu lại.\n")
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, x, sr, subtype="PCM_24")
            print(f"         ✓ đã lưu {path}")

            r2 = input("         [Enter]=giữ   r=thu lại  > ").strip().lower()
            if r2 == "r":
                path.unlink()
                print()
                continue
            done.append({"name": name, "group": group, "seconds": secs, **stats})
            print()
            break

    _write_meta(root, device, info, sr, done, skipped)
    print("=" * 74)
    print(f"  XONG. {len(done)} bản thu mới, {len(skipped)} bỏ qua.")
    print(f"  Metadata: {root/'session.json'}")
    print()
    print("  CÒN MỘT VIỆC: mở recordings/README.txt và điền")
    print("    - loại đàn (thép acoustic / nylon classical / electric)")
    print("    - số cent lệch của từng dây theo tuner tham chiếu, TRƯỚC khi thu")
    print("=" * 74)


def _write_meta(root, device, info, sr, done, skipped):
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "device_index": int(device),
        "device_name": info["name"],
        "device_native_samplerate": float(info["default_samplerate"]),
        "samplerate": sr,
        "channels": 1,
        "dtype_captured": "float32",
        "file_subtype": "PCM_24",
        "processing": "none (PortAudio/CoreAudio raw, no AGC/NR/normalize)",
        "takes": done,
        "skipped": skipped,
        # điền tay:
        "guitar_type": "TODO: acoustic-steel | classical-nylon | electric",
        "reference_tuner": "TODO: tên app",
        "string_offsets_cents": {"E2": None, "A2": None, "D3": None,
                                 "G3": None, "B3": None, "E4": None},
    }
    (root / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "device      : {}\n"
            "samplerate  : {}\n"
            "bit_depth   : 24\n"
            "guitar      : (điền: acoustic thép / classical nylon / electric)\n"
            "tuned_with  : (điền: tên app tuner tham chiếu)\n"
            "offsets     : (điền, ví dụ) E2 -2c, A2 +1c, D3 0c, G3 -3c, B3 +2c, E4 -1c\n"
            .format(info["name"], sr))


if __name__ == "__main__":
    main()
