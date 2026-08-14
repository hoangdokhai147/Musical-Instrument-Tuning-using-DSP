"""
Framer — tích luỹ chunk kích thước bất kỳ, phát frame cố định theo nhịp hop.

ĐÂY LÀ KHỐI ĐANG THIẾU khiến engine chưa cắm được vào microphone thật. Nó tách
ba đại lượng mà bản đầu gộp vào một tham số của Tuner.process():

    chunk   hệ điều hành áp đặt. 256, 512, 1024 tuỳ thiết bị và driver, và có
            thể ĐỔI GIỮA CÁC LẦN GỌI. Ứng dụng không kiểm soát được.
    frame   ràng buộc DSP quyết định: W + tau_max + 1 = 4836 mẫu. Cố định.
    hop     trải nghiệm người dùng quyết định: kim nhúc nhích bao lâu một lần.
            1024 mẫu = 21.3 ms = 47 lần/giây.

Ba thứ đó độc lập với nhau. Ép chúng bằng nhau — như bản đầu buộc phải làm — có
nghĩa là hoặc kim giật 10 lần/giây, hoặc engine không nhận nổi audio từ mic.

THỨ TỰ: LỌC TRƯỚC, CẮT FRAME SAU
    StreamingFIR phải thấy luồng audio ĐÚNG MỘT LẦN, liên tục. Nếu cắt frame
    trước rồi lọc từng frame, bộ lọc sẽ thấy dữ liệu chồng lấn và state của nó
    trở nên vô nghĩa.

    Đây không phải lo xa. Harness đo golden ban đầu nạp cửa sổ chồng lấn vào FIR
    có state, và đo được: sai số đỉnh 122.5% biên độ, 464/465 frame lệch, f0
    lệch tới 12.2 cent. Đúng len(h)-1 = 126 mẫu đầu mỗi frame bị hỏng, vì đuôi
    FIR được nối vào từ một vị trí không liền mạch trong tín hiệu.
"""

import numpy as np


class Framer:
    """
    Cửa sổ trượt: giữ frame_len mẫu gần nhất, phát một bản sao sau mỗi hop mẫu.

    Frame ĐẦU TIÊN chỉ phát khi đã đủ frame_len mẫu — trước đó chưa có gì để đo.
    Từ frame thứ hai trở đi, mỗi hop mẫu mới lại phát một frame, và các frame
    chồng lấn nhau frame_len - hop mẫu.
    """

    def __init__(self, frame_len, hop):
        if not (1 <= hop <= frame_len):
            raise ValueError(f"cần 1 <= hop <= frame_len, nhận hop={hop} "
                             f"frame_len={frame_len}")
        self.frame_len = int(frame_len)
        self.hop = int(hop)
        self._buf = np.zeros(self.frame_len, dtype=np.float64)
        self._countdown = self.frame_len     # còn thiếu bao nhiêu mẫu nữa thì phát
        self.total_in = 0                    # để chẩn đoán và kiểm tra bất biến
        self.total_out = 0

    def push(self, x):
        """
        Nạp chunk dài BẤT KỲ (kể cả 0, kể cả dài hơn frame_len).
        Trả list gồm 0, 1 hoặc nhiều frame — mỗi frame là mảng frame_len mẫu.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self.total_in += len(x)
        out = []
        i = 0
        while i < len(x):
            take = min(self._countdown, len(x) - i)
            self._shift_in(x[i:i + take])
            self._countdown -= take
            i += take
            if self._countdown == 0:
                # Sao chép: người gọi có thể giữ frame lại, còn _buf sẽ bị ghi đè
                # ở lần push sau. 38 KB mỗi frame ở 47 fps = 1.8 MB/s — không đáng
                # kể so với 24 MB/frame mà difference_function đã tạo ra.
                out.append(self._buf.copy())
                self.total_out += 1
                self._countdown = self.hop
        return out

    def _shift_in(self, seg):
        n = len(seg)
        if n == 0:
            return
        if n >= self.frame_len:
            self._buf[:] = seg[-self.frame_len:]
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = seg

    def reset(self):
        self._buf[:] = 0.0
        self._countdown = self.frame_len
        self.total_in = 0
        self.total_out = 0

    @property
    def latency_samples(self):
        """Số mẫu phải chờ trước khi có frame ĐẦU TIÊN."""
        return self.frame_len
