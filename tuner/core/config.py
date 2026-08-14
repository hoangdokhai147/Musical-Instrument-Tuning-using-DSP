"""
Tham số của engine. Không có hành vi — chỉ số, ràng buộc, và lời giải thích vì sao.

FROZEN DATACLASS, không phải class thường. Hai cạm bẫy của bản cũ:

  1. `Config.tau_min` truy cập trên CLASS trả về đối tượng `property`, không phải
     số. Chỉ hoạt động khi truy cập trên INSTANCE. Đây đúng là lỗi đã có trong
     dsp_core.py và bị mang sang: PitchEngine(Config) ném TypeError khó hiểu
     "unsupported operand type(s) for /: 'property' and 'float'".

  2. `Config.a4 = 442` đổi giá trị cho MỌI instance, vì đó là biến class chứ
     không phải biến instance. Một màn hình cài đặt sửa A4 sẽ vô tình đổi luôn
     cho mọi engine đang chạy.

frozen=True chặn cả hai: gán vào instance ném FrozenInstanceError, và muốn có
biến thể thì dùng dataclasses.replace(cfg, a4=442) — tường minh, cục bộ, an toàn.

VALIDATE CHẠY TRONG __post_init__, không phải một hàm phải nhớ gọi. Config sai
không tạo ra được, nên không có trạng thái "đã tạo nhưng chưa kiểm tra".
"""

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
    W: int           = 4096      # cửa sổ tham chiếu (mẫu). Xem ghi chú (A) bên dưới.
    hop: int         = 1024      # bước nhảy giữa hai lần phát hiện (21.3 ms @48k)
    fc_lp: float     = 2000.0    # cắt lowpass (Hz). Xem ghi chú (B).
    lp_taps: int     = 127       # bậc FIR (lẻ -> đối xứng -> pha tuyến tính)
    thresh: float    = 0.15      # ngưỡng tuyệt đối CMNDF (YIN dùng 0.10-0.15)
    a4: float        = 440.0

    # --- các ngưỡng dưới đây hiệu chuẩn từ 25 bản thu guitar thật, xem mục (C) ---
    rms_gate: float    = 0.005   # -46 dBFS. Nền phòng đo được -58.8 dBFS -> còn 13 dB biên.
    min_conf: float    = 0.85    # = 1 - thresh. Xem (C1).
    sub_margin: float  = 18.0    # dB. Chặn subharmonic. Xem (C2).
    hold_frames: int   = 23      # ~0.5 s. Giữ trạng thái qua khoảng mất tín hiệu ngắn. Xem (C3).
    med_size: int      = 15      # cửa sổ median
    ema_slow: float    = 0.10    # alpha khi ổn định
    ema_fast: float    = 0.90    # alpha khi đang vặn khoá
    ema_jump_c: float  = 25.0    # cent. Vượt ngưỡng này -> chuyển sang ema_fast.
    in_tune_c: float   = 5.0     # dung sai hiển thị "đúng cao độ"

    # (A) VÌ SAO W = 4096?
    #     Ràng buộc dưới: d(τ) chỉ có nghĩa khi cửa sổ tham chiếu dài hơn chu kỳ dài nhất.
    #     τ_max = fs/f_min = 739 mẫu, nên W = 4096 cho 5.5 chu kỳ của 65 Hz — thoải mái.
    #     Ràng buộc trên: latency. buffer = W + τ_max = 4835 mẫu = 100.7 ms.
    #     Đo được: W=2048 (58 ms) cũng đạt ±1 cent, chỉ kém hơn khi nhiễu nặng
    #     (SNR 10 dB: MAE 1.07 vs 0.70 cent). Giảm xuống 2048 nếu cần latency thấp.
    #
    # (B) VÌ SAO fc = 2000 Hz?
    #     Chỉ cần > f_max = 1100 Hz đủ margin để không suy giảm f0 cao nhất.
    #     Đo được |H(1100 Hz)| = −0.01 dB, |H(3000 Hz)| = −67 dB.
    #     Vì KHÔNG hạ mẫu nên bộ lọc này KHÔNG phải anti-alias — nó chỉ có một việc:
    #     bỏ công suất nhiễu ngoài dải. Yêu cầu vì thế lỏng hơn nhiều so với bản cũ.
    #
    #     Bộ lọc này KHÔNG thay thế được bằng W dài hơn. Đo được ở SNR 10 dB:
    #         W=2048 không lọc -> 200/259 lỗi octave
    #         W=8192 không lọc -> 197/259 lỗi octave   (gấp 4 lần W, gần như vô ích)
    #         W=2048 CÓ lọc    ->   0/259 lỗi octave
    #     Lý do: nhiễu trắng cộng vào d(τ) một lượng ≈ 2σ²W ở MỌI τ > 0. Tăng W làm
    #     tăng cả số hạng tín hiệu lẫn số hạng nhiễu theo cùng tỉ lệ, nên tỉ số không
    #     đổi. Chỉ giới hạn băng thông (giảm σ²) mới cải thiện được.
    #
    # (C) CÁC NGƯỠNG HIỆU CHUẨN TỪ BẢN THU THẬT (25 take, guitar thép, mic MacBook)
    #
    #     C1. min_conf 0.50 -> 0.85.  find_period chấp nhận lag khi d' < thresh = 0.15,
    #         tức conf > 0.85. Nhưng cổng cũ chỉ loại khi conf < 0.50, nên frame ĐÃ
    #         TRƯỢT ngưỡng YIN (rơi vào nhánh fallback argmin) vẫn được báo là hợp lệ.
    #         Đặt bằng 1 - thresh làm hai ngưỡng nhất quán. Đo: octave 3.8% -> 1.8%.
    #
    #     C2. sub_margin: chặn subharmonic bằng kiểm chứng phổ. Ở frame lỗi, engine báo
    #         109.65 Hz trong khi công suất tại đó là -117.5 dB còn tần số thật
    #         (329.6 Hz) là 0 dB — nó báo một chu kỳ KHÔNG CÓ NĂNG LƯỢNG. confidence
    #         không bắt được vì subharmonic vẫn thực sự tuần hoàn. Đo: 3.8% -> 2.5%,
    #         kết hợp với C1 -> 0.7%.  Phân bố p(f0)-max(p(2f0),p(3f0)):
    #         frame đúng trung vị +3.1 dB, frame lỗi trung vị -15.7 dB.
    #
    #     C3. hold_frames — THAY ĐỔI QUAN TRỌNG NHẤT, và không phải chuyện tham số.
    #         Bản cũ gọi reset() ngay khi MỘT frame rơi dưới gate. Nốt đang tắt dần
    #         nhấp nháy quanh ngưỡng -> bộ đệm median bị xoá liên tục, không bao giờ
    #         kịp đầy 15 mẫu. Vì thế tăng med_size hay giảm ema_slow đều gần như VÔ ÍCH:
    #             quét med_size 1->21  : jitter p95 chỉ giảm 24.8 -> 23.3 cent
    #             quét ema_slow .40->.03: jitter p95 chỉ giảm 25.3 -> 18.3 cent
    #         Chỉ cần ngừng reset ngay lập tức:
    #             hold=0 (cũ) -> jitter p95 tệ nhất 25.3 cent
    #             hold=5      -> 12.8 cent
    #             hold=23     -> 12.8 cent, và A2 từ 25.3 xuống 0.6 cent
    #         Kết hợp với rms_gate -46 dBFS: tệ nhất 5.9 cent, năm trong sáu dây < 1.2.
    #         Bám nốt mới KHÔNG chậm đi (đo trên notechange: 21 ms, không đổi) vì
    #         cơ chế ema_jump_c vẫn kích hoạt ema_fast khi cao độ nhảy.

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
        need(self.a4 > 0, f"a4 ({self.a4}) phải dương")
