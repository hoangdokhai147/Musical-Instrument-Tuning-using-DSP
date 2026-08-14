"""
golden_vectors.py — BƯỚC 0 của kế hoạch refactor: đóng băng hành vi hiện tại.

    python tests/golden_vectors.py freeze    # sinh tests/golden/vectors.json
    python tests/golden_vectors.py check     # so hành vi hiện tại với bản đã đóng băng
    python tests/golden_vectors.py check -v  # in chi tiết mọi sai lệch

VÌ SAO CẦN
  Refactor phải BẢO TOÀN hành vi. Không có cách chứng minh điều đó thì "refactor"
  và "vô tình đổi thuật toán" là hai việc không phân biệt được. File này là bằng
  chứng: nếu check còn xanh thì hành vi chưa đổi, bất kể code đã bị xáo trộn thế nào.

TẠI SAO FREEZE VÀ CHECK NẰM CHUNG MỘT FILE
  Đoạn code đo CHÍNH LÀ hợp đồng. Tách làm hai file thì hai bên sẽ trôi khỏi nhau,
  và lúc đó check xanh không còn nghĩa lý gì.

SO SÁNH TUYỆT ĐỐI, KHÔNG DUNG SAI
  Mọi thứ ở đây tất định: cùng seed, cùng input, cùng code -> cùng bit. Bước 1-3 của
  kế hoạch chỉ DI CHUYỂN code nên phải khớp bit-exact. Dung sai chỉ nới ở bước 4-5,
  nơi hành vi thay đổi có chủ đích — và khi đó phải freeze lại một cách CÓ Ý THỨC,
  kèm ghi chú vì sao.

  Ngoại lệ duy nhất: REL_TOL cho khác biệt phiên bản numpy/BLAS ở ULP cuối.
"""

import argparse, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

# ---------------------------------------------------------------------------
# ĐIỂM NỐI DUY NHẤT TỚI ENGINE.
# Khi cấu trúc engine đổi, CHỈ SỬA KHỐI NÀY. Nếu phải sửa chỗ khác trong file thì
# nghĩa là API công khai đã đổi, và đó là thông tin đáng biết.
#
# Bước 1 (tách dsp_simple.py -> gói tuner/): chỉ khối này thay đổi. Bốn tên dưới
# đây là toàn bộ bề mặt mà bộ test chạm tới.
# ---------------------------------------------------------------------------
from tuner.core.config import Config
from tuner.core.detector import PitchDetector
from tuner.core.filters import design_lowpass
from tuner.engine.engine import Tuner

VECTORS = HERE / "golden" / "vectors.json"
RECORDINGS = PROJ / "recordings"
REL_TOL = 1e-12          # chỉ để hấp thụ khác biệt ULP giữa các bản numpy/BLAS


# ===========================================================================
# SINH TÍN HIỆU — tất định, mô tả bằng SPEC chứ không nhúng mảng mẫu.
# Bản port Dart phải tự sinh lại được từ spec; chính việc đó phát hiện sai lệch
# trong sin/hanning/sinc ở ULP cuối mà một mảng nhúng sẵn sẽ che mất.
# ===========================================================================

def synth(f0, n, fs, harmonics, noise, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for k in range(1, harmonics + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    y *= 0.5 / np.max(np.abs(y))
    if noise:
        y = y + noise * rng.standard_normal(n)
    return y


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


# ===========================================================================
# BA NHÓM PHÉP ĐO
# ===========================================================================

def measure_synthetic(cfg):
    """Sweep liên tục — nhóm đã bắt được cả ba bug trong audit Volume I.
    Test ở các nốt chuẩn thì cả ba đều lọt, nên nhóm này là BẮT BUỘC."""
    det = PitchDetector(cfg)
    lp = design_lowpass(cfg.fs, cfg.fc_lp, cfg.lp_taps)
    n = cfg.buf_len + 400
    out = []
    for name, harm, noise in (("sine", 1, 0.0), ("harm6", 6, 0.005)):
        for i, f0 in enumerate(np.arange(65.0, 1100.0, 1.0)):
            x = np.convolve(synth(f0, n, cfg.fs, harm, noise, i), lp, mode="same")
            r = det.detect(x)
            out.append({"id": f"{name}_{f0:.1f}", "f0_true": float(f0),
                        "harmonics": harm, "noise": noise, "seed": i,
                        "f0": r.f0, "conf": r.confidence, "sub_db": r.sub_db})
    return out


def measure_detect_files(cfg, rels):
    """Chạy detector THUẦN trên một frame cố định của từng file (bỏ 200 ms đầu).
    Bỏ 200 ms vì mọi bản thu đều có onset transient — loa nảy, dây gảy, phòng dội."""
    det = PitchDetector(cfg)
    lp = design_lowpass(cfg.fs, cfg.fc_lp, cfg.lp_taps)
    skip = int(0.20 * cfg.fs)
    out = {}
    for rel in rels:
        p = RECORDINGS / rel
        if not p.exists():
            continue
        x, sr = load_wav(p)
        if sr != cfg.fs or len(x) < skip + cfg.buf_len:
            continue
        r = det.detect(np.convolve(x[skip:skip + cfg.buf_len + 400], lp, mode="same"))
        out[rel] = {"sr": sr, "n": len(x),
                    "f0": r.f0, "conf": r.confidence, "sub_db": r.sub_db}
    return out


def measure_stream_files(cfg, rels):
    """Chạy Tuner ĐẦY ĐỦ trên trọn file — bắt cả state machine, smoothing, hold,
    overlap-save, framing. Lưu chuỗi từng frame chứ không chỉ số tổng hợp: tổng
    hợp có thể giữ nguyên trong khi từng frame đã đổi.

    Nạp qua push() với chunk 1024 mẫu — đúng cách ứng dụng thật sẽ dùng.

    BẢN ĐẦU (bước 0-3) nạp cửa sổ CHỒNG LẤN dài buf_len vào một FIR CÓ STATE:
        for i in range(0, len(x)-buf_len, hop): tu.process(x[i:i+buf_len])
    Mỗi lần nạp 4836 mẫu nhưng chỉ tiến 1024, nên 3812 mẫu bị lọc lại, và đuôi
    FIR được nối vào từ một vị trí không liền mạch. Đo được: 126 mẫu đầu mỗi
    frame sai (đúng bằng len(h)-1), sai số đỉnh 122.5% biên độ, 464/465 frame
    lệch, f0 lệch tới 12.2 cent.

    Đó là lỗi của HARNESS, không phải của engine — các phép đo chất lượng đã công
    bố (jitter 0.37-4.93 cent, octave 0.10%) đều lọc cả file một lần rồi mới cắt
    frame, tức đúng. Nhưng vector đóng băng ở bước 0 thì có dính, nên bước 4 phải
    freeze lại."""
    out = {}
    for rel in rels:
        p = RECORDINGS / rel
        if not p.exists():
            continue
        x, sr = load_wav(p)
        if sr != cfg.fs:
            continue
        tu = Tuner(cfg)
        states, f0s = [], []
        for i in range(0, len(x), 1024):
            for o in tu.push(x[i:i + 1024]):
                states.append(o["state"])
                f0s.append(o.get("f0"))
        det = [v for v in f0s if v is not None]
        out[rel] = {
            "n_frames": len(states),
            "states": "".join(s[0] for s in states),   # D/S/U/B — gọn và đọc được
            "f0": f0s,
            "summary": {
                "n_detecting": len(det),
                "median_f0": float(np.median(det)) if det else None,
            },
        }
    return out


ALL_REAL = [
    "calib/tone_82.wav", "calib/tone_110.wav", "calib/tone_220.wav",
    "calib/tone_440.wav", "calib/tone_sweep.wav", "calib/silence_room.wav",
    "steady/guitar_E2_open.wav", "steady/guitar_A2_open.wav", "steady/guitar_D3_open.wav",
    "steady/guitar_G3_open.wav", "steady/guitar_B3_open.wav", "steady/guitar_E4_open.wav",
    "steady/guitar_A2_flat30.wav",
    "transient/guitar_E2_plucks.wav", "transient/guitar_E2_hard.wav",
    "transient/guitar_E2_soft.wav",
    "hard/guitar_harmonic12_E4.wav", "hard/guitar_chromatic_highE.wav",
    "hard/guitar_notechange.wav", "hard/guitar_E2_decay.wav",
    "env/guitar_E2_noisy.wav", "env/guitar_E2_quiet.wav",
    "reject/guitar_two_strings.wav", "reject/guitar_chord_Em.wav",
    "reject/guitar_percussive.wav",
]
DIGITAL = sorted(str(p.relative_to(RECORDINGS)) for p in (RECORDINGS / "digital").glob("*.wav")) \
          if (RECORDINGS / "digital").exists() else []


def config_snapshot(cfg):
    """Vector chỉ có nghĩa với ĐÚNG config đã sinh ra chúng. Đổi config mà check
    vẫn chạy thì kết quả 'fail' sẽ bị đọc nhầm thành hồi quy DSP."""
    keys = [k for k in dir(cfg)
            if not k.startswith("_") and isinstance(getattr(cfg, k), (int, float))]
    return {k: getattr(cfg, k) for k in sorted(keys)}


def build():
    cfg = Config()
    t0 = time.time()
    print("  [1/3] sweep tổng hợp (2070 điểm)...", flush=True)
    syn = measure_synthetic(cfg)
    print(f"        {len(syn)} vector  ({time.time()-t0:.0f}s)", flush=True)
    print("  [2/3] detect trên file (đối chứng số + tone)...", flush=True)
    dets = measure_detect_files(cfg, DIGITAL + ALL_REAL)
    print(f"        {len(dets)} file  ({time.time()-t0:.0f}s)", flush=True)
    print("  [3/3] Tuner đầy đủ theo hop trên 25 bản thu...", flush=True)
    strm = measure_stream_files(cfg, ALL_REAL)
    print(f"        {len(strm)} file  ({time.time()-t0:.0f}s)", flush=True)
    src = b"".join(sorted(p.read_bytes() for p in (PROJ / "tuner").rglob("*.py")))
    return {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(src).hexdigest(),
            "note": ("source_sha256 chỉ để truy vết. Refactor SẼ làm nó đổi — "
                     "đó là chuyện bình thường, không phải lỗi."),
            "config": config_snapshot(cfg),
        },
        "synthetic": syn,
        "detect_files": dets,
        "stream_files": strm,
    }


# ===========================================================================
# SO SÁNH
# ===========================================================================

def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if not (np.isfinite(a) and np.isfinite(b)):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return abs(a - b) <= REL_TOL * max(1.0, abs(b))


def check(verbose):
    if not VECTORS.exists():
        sys.exit(f"Chưa có {VECTORS}. Chạy 'freeze' trước.")
    ref = json.loads(VECTORS.read_text())
    cfg = Config()

    cur_cfg = config_snapshot(cfg)
    if cur_cfg != ref["meta"]["config"]:
        diff = {k: (ref["meta"]["config"].get(k), cur_cfg.get(k))
                for k in set(cur_cfg) | set(ref["meta"]["config"])
                if ref["meta"]["config"].get(k) != cur_cfg.get(k)}
        print("✗ CONFIG ĐÃ ĐỔI — vector cũ không còn dùng để so được.")
        for k, (o, n) in sorted(diff.items()):
            print(f"    {k}: {o} -> {n}")
        print("\n  Nếu đổi config là CÓ CHỦ ĐÍCH thì chạy lại 'freeze' và ghi vào commit vì sao.")
        return 2

    fails = []

    print("  [1/3] sweep tổng hợp...", flush=True)
    for a, b in zip(measure_synthetic(cfg), ref["synthetic"]):
        for k in ("f0", "conf", "sub_db"):
            if not close(a[k], b[k]):
                fails.append((f"synthetic/{b['id']}", k, b[k], a[k]))

    print("  [2/3] detect trên file...", flush=True)
    cur = measure_detect_files(cfg, DIGITAL + ALL_REAL)
    for rel, b in ref["detect_files"].items():
        a = cur.get(rel)
        if a is None:
            fails.append((f"detect/{rel}", "missing", "có", "thiếu file")); continue
        for k in ("f0", "conf", "sub_db"):
            if not close(a[k], b[k]):
                fails.append((f"detect/{rel}", k, b[k], a[k]))

    print("  [3/3] Tuner đầy đủ...", flush=True)
    cur = measure_stream_files(cfg, ALL_REAL)
    for rel, b in ref["stream_files"].items():
        a = cur.get(rel)
        if a is None:
            fails.append((f"stream/{rel}", "missing", "có", "thiếu file")); continue
        if a["states"] != b["states"]:
            n = sum(1 for p, q in zip(a["states"], b["states"]) if p != q)
            n += abs(len(a["states"]) - len(b["states"]))
            fails.append((f"stream/{rel}", "states", f"{len(b['states'])} frame",
                          f"{n} frame khác"))
        for i, (p, q) in enumerate(zip(a["f0"], b["f0"])):
            if not close(p, q):
                fails.append((f"stream/{rel}", f"f0[{i}]", q, p))
                if not verbose:
                    break

    print()
    if not fails:
        print("=" * 78)
        print("  ✓ KHỚP HOÀN TOÀN — hành vi chưa đổi.")
        print(f"    {len(ref['synthetic'])} vector tổng hợp · "
              f"{len(ref['detect_files'])} file detect · "
              f"{len(ref['stream_files'])} file stream")
        print(f"    vector đóng băng lúc {ref['meta']['created']}")
        print("=" * 78)
        return 0

    print("=" * 78)
    print(f"  ✗ LỆCH {len(fails)} chỗ")
    print("=" * 78)
    for where, what, want, got in (fails if verbose else fails[:25]):
        print(f"    {where:<42} {what:<10} mong đợi {want}  nhận {got}")
    if not verbose and len(fails) > 25:
        print(f"    ... còn {len(fails)-25} chỗ nữa (chạy với -v để xem hết)")
    print()
    print("  Nếu thay đổi là CÓ CHỦ ĐÍCH: chạy 'freeze' lại và ghi vào commit vì sao.")
    print("  Nếu KHÔNG: refactor vừa rồi đã làm đổi hành vi. Đó chính là điều test này")
    print("  sinh ra để bắt.")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "check"])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.mode == "freeze":
        if VECTORS.exists():
            print(f"⚠  {VECTORS} đã tồn tại — sẽ GHI ĐÈ.")
            if input("   Chắc chưa? [y/N] ").strip().lower() != "y":
                return 1
        print("Đang sinh golden vectors...")
        data = build()
        VECTORS.parent.mkdir(parents=True, exist_ok=True)
        VECTORS.write_text(json.dumps(data, indent=1))
        kb = VECTORS.stat().st_size / 1024
        print(f"\n✓ Đã ghi {VECTORS}  ({kb:.0f} KB)")
        print(f"  {len(data['synthetic'])} vector tổng hợp · "
              f"{len(data['detect_files'])} file detect · "
              f"{len(data['stream_files'])} file stream")
        print("\n  HÃY COMMIT FILE NÀY TRƯỚC KHI SỬA BẤT CỨ THỨ GÌ.")
        return 0

    print("Đang so hành vi hiện tại với bản đã đóng băng...")
    return check(a.verbose)


if __name__ == "__main__":
    sys.exit(main())
