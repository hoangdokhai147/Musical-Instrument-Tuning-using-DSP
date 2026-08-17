from dataclasses import dataclass

import numpy as np


# =============================================================================
# 0. CẤU HÌNH
# =============================================================================

@dataclass(frozen=True)
class Config:
    fs: float        = 48000.0   # sample rate (Hz). KHÔNG hạ mẫu -> không có fs_eff.
    f_min: float     = 65.0      # C2  — dưới E2 của guitar, phủ cả cello/viola
    f_max: float     = 1100.0    # ~C#6 — phủ trọn cần đàn guitar và dây buông violin (E5=659)
    W: int           = 4096      # cửa sổ tham chiếu (mẫu).
    hop: int         = 1024      # bước nhảy giữa hai lần phát hiện (21.3 ms @48k)
    fc_lp: float     = 2000.0    # cắt lowpass (Hz).
    lp_taps: int     = 127       # bậc FIR (lẻ -> đối xứng -> pha tuyến tính)
    thresh: float    = 0.15      # ngưỡng tuyệt đối CMNDF (YIN dùng 0.10-0.15)
    a4: float        = 440.0

    # --- các ngưỡng dưới đây hiệu chuẩn từ 25 bản thu guitar thật ---
    rms_gate: float    = 0.005   # -46 dBFS. Nền phòng đo được -58.8 dBFS -> còn 13 dB biên.
    min_conf: float    = 0.85    # = 1 - thresh.
    sub_margin: float  = 18.0    # dB. Chặn subharmonic.
    hold_frames: int   = 23      # ~0.5 s. Giữ trạng thái qua khoảng mất tín hiệu ngắn.
    med_size: int      = 15      # cửa sổ median
    ema_slow: float    = 0.10    # alpha khi ổn định
    ema_fast: float    = 0.90    # alpha khi đang vặn khoá
    ema_jump_c: float  = 25.0    # cent. Vượt ngưỡng này -> chuyển sang ema_fast.
    in_tune_c: float   = 5.0     # vào IN_TUNE khi |cent| <= giá trị này

    # --- lớp HIỂN THỊ: thẩm mỹ, không phải DSP. Đổi thoải mái, không ảnh hưởng độ đo. ---
    in_tune_release_c: float = 8.0    # chỉ RA khỏi IN_TUNE khi vượt — chống chớp đèn
    note_hyst_c: float       = 15.0   # giữ nốt cũ tới 50+15 cent — chống nhấp nháy tên nốt
    acquire_frames: int      = 5      # số frame ổn định trước khi ACQUIRING -> LOCKED
    display_hold_ms: float   = 800.0  # giữ số đọc bao lâu sau khi mất tín hiệu

    @property
    def tau_min(self): return int(self.fs / self.f_max)          # 43
    @property
    def tau_max(self): return int(np.ceil(self.fs / self.f_min)) # 739
    @property
    def buf_len(self): return self.W + self.tau_max + 1          # 4836

    # -------------------------------------------------------------------------
    # KIỂM TRA RÀNG BUỘC
    #
    # Mỗi dòng dưới đây tương ứng một cách config có thể sai mà engine vẫn CHẠY,
    # chỉ là cho ra rác. Ví dụ f_max = 30000 với fs = 48000 cho tau_min = 1, và
    # find_period sẽ dò từ lag 1 — không ném lỗi, chỉ trả về số vô nghĩa.
    # Thà không tạo được Config còn hơn debug một engine im lặng cho kết quả sai.
    # -------------------------------------------------------------------------

    def __post_init__(self):
        def need(cond, msg):
            if not cond:
                raise ValueError(f"Config không hợp lệ: {msg}")

        need(self.fs > 0, f"fs phải dương, nhận {self.fs}")
        need(0 < self.f_min < self.f_max,
             f"cần 0 < f_min < f_max, nhận f_min={self.f_min} f_max={self.f_max}")
        need(self.f_max < self.fs / 2,
             f"f_max ({self.f_max}) phải nhỏ hơn Nyquist ({self.fs/2})")

        # tau_min >= 2 để parabolic_interp còn điểm bên trái mà nội suy
        need(self.tau_min >= 2,
             f"f_max quá cao: tau_min = {self.tau_min}, cần >= 2 để nội suy được")
        # W phải phủ được chu kỳ dài nhất, nếu không d(tau_max) vô nghĩa
        need(self.W >= self.tau_max,
             f"W ({self.W}) phải >= tau_max ({self.tau_max}), nếu không cửa sổ "
             f"tham chiếu ngắn hơn chu kỳ cần đo")
        need(1 <= self.hop <= self.buf_len,
             f"hop ({self.hop}) phải trong [1, buf_len={self.buf_len}]")

        need(self.f_max < self.fc_lp < self.fs / 2,
             f"cần f_max < fc_lp < Nyquist, nhận f_max={self.f_max} "
             f"fc_lp={self.fc_lp} Nyquist={self.fs/2} — lowpass đang cắt vào dải f0")
        # design_lowpass tự tăng lên số lẻ; bắt lỗi ở đây để không có bất ngờ thầm lặng
        need(self.lp_taps % 2 == 1, f"lp_taps ({self.lp_taps}) phải LẺ để pha tuyến tính")
        need(self.lp_taps >= 3, f"lp_taps ({self.lp_taps}) quá nhỏ")

        need(0 < self.thresh < 1, f"thresh ({self.thresh}) phải trong (0, 1)")
        need(0 <= self.rms_gate < 1, f"rms_gate ({self.rms_gate}) phải trong [0, 1)")
        need(0 <= self.min_conf <= 1, f"min_conf ({self.min_conf}) phải trong [0, 1]")
        need(self.sub_margin > 0, f"sub_margin ({self.sub_margin}) phải dương")
        need(self.hold_frames >= 0, f"hold_frames ({self.hold_frames}) không được âm")

        need(self.med_size >= 1, f"med_size ({self.med_size}) phải >= 1")
        # median cửa sổ CHẴN lấy trung bình hai phần tử giữa -> sinh giá trị chưa
        # từng được đo. Với f0 thì đó là nội suy lén lút, không phải lọc outlier.
        need(self.med_size % 2 == 1, f"med_size ({self.med_size}) phải LẺ")

        need(0 < self.ema_slow <= self.ema_fast <= 1,
             f"cần 0 < ema_slow <= ema_fast <= 1, nhận {self.ema_slow} / {self.ema_fast}")
        need(self.ema_jump_c > 0, f"ema_jump_c ({self.ema_jump_c}) phải dương")
        need(self.in_tune_c > 0, f"in_tune_c ({self.in_tune_c}) phải dương")
        need(self.in_tune_release_c >= self.in_tune_c,
             f"in_tune_release_c ({self.in_tune_release_c}) phải >= in_tune_c "
             f"({self.in_tune_c}), nếu không hysteresis đảo chiều và đèn chớp còn tệ hơn")
        need(self.note_hyst_c >= 0, f"note_hyst_c ({self.note_hyst_c}) không được âm")
        need(self.note_hyst_c < 50, f"note_hyst_c ({self.note_hyst_c}) phải < 50 — "
             f"lớn hơn nửa cung thì không bao giờ đổi được nốt")
        need(self.acquire_frames >= 1, f"acquire_frames ({self.acquire_frames}) phải >= 1")
        need(self.display_hold_ms >= 0, f"display_hold_ms ({self.display_hold_ms}) không được âm")
        need(self.a4 > 0, f"a4 ({self.a4}) phải dương")
