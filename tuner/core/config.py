"""Tham số của engine. Không có hành vi — chỉ số và lời giải thích vì sao."""

import numpy as np


# =============================================================================
# 0. CẤU HÌNH
# =============================================================================

class Config:
    fs        = 48000.0   # sample rate (Hz). KHÔNG hạ mẫu -> không có fs_eff.
    f_min     = 65.0      # C2  — dưới E2 của guitar, phủ cả cello/viola
    f_max     = 1100.0    # ~C#6 — phủ trọn cần đàn guitar và dây buông violin (E5=659)
    W         = 4096      # cửa sổ tham chiếu (mẫu). Xem ghi chú (A) bên dưới.
    hop       = 1024      # bước nhảy giữa hai lần phát hiện (21.3 ms @48k)
    fc_lp     = 2000.0    # cắt lowpass (Hz). Xem ghi chú (B).
    lp_taps   = 127       # bậc FIR (lẻ -> đối xứng -> pha tuyến tính)
    thresh    = 0.15      # ngưỡng tuyệt đối CMNDF (YIN dùng 0.10-0.15)
    a4        = 440.0

    # --- các ngưỡng dưới đây hiệu chuẩn từ 25 bản thu guitar thật, xem mục (C) ---
    rms_gate    = 0.005   # -46 dBFS. Nền phòng đo được -58.8 dBFS -> còn 13 dB biên.
    min_conf    = 0.85    # = 1 - thresh. Xem (C1).
    sub_margin  = 18.0    # dB. Chặn subharmonic. Xem (C2).
    hold_frames = 23      # ~0.5 s. Giữ trạng thái qua khoảng mất tín hiệu ngắn. Xem (C3).
    med_size    = 15      # cửa sổ median
    ema_slow    = 0.10    # alpha khi ổn định
    ema_fast    = 0.90    # alpha khi đang vặn khoá
    ema_jump_c  = 25.0    # cent. Vượt ngưỡng này -> chuyển sang ema_fast.
    in_tune_c   = 5.0     # dung sai hiển thị "đúng cao độ"

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


# =============================================================================
# 1. FIR LOWPASS — window method (Proakis Ch 10)
# =============================================================================
