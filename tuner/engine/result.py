"""
Hợp đồng giữa engine và giao diện.

MỘT QUY TẮC DUY NHẤT, và mọi thứ khác suy ra từ nó:

    Các trường ĐO ĐƯỢC khác None KHI VÀ CHỈ KHI status ∈ {ACQUIRING, LOCKED, HOLDING}.

Nhờ vậy UI không phải đoán. Không có "chuỗi rỗng nghĩa là chưa có", không có
"f0 = 0 nghĩa là im lặng", không có `.get()` phòng thủ ở mọi chỗ. Một câu lệnh
switch trên status là đủ.

VÌ SAO CÓ HOLDING
  Người dùng vặn khoá rồi gảy lại. Giữa hai lần gảy, tín hiệu tắt. Nếu màn hình
  nhấp về rỗng mỗi lần đó thì dùng rất khó chịu. Nhưng nếu engine cứ báo LOCKED
  trong lúc thật ra chẳng nghe thấy gì thì nó đang NÓI DỐI.

  HOLDING nói đúng sự thật: "đây là số đọc gần nhất, nó đã cũ stale_ms mili giây,
  ngay lúc này không có tín hiệu". UI làm mờ đi là hợp lý. Sau display_hold_ms
  thì chuyển hẳn sang SILENT và xoá màn hình.

VÌ SAO TÁCH note_name VÀ octave
  Mockup UI render chúng ở hai cỡ chữ khác nhau — chữ "A" rất lớn, số "4" nhỏ ở
  trên. Trả về chuỗi "A4" thì UI phải tự cắt, mà cắt chuỗi là chỗ dễ sai.

VÌ SAO CÓ CẢ raw_frequency_hz
  Khi kim rung, câu hỏi đầu tiên là "do detector hay do làm mượt". Không có số
  thô thì không trả lời được. Chi phí bằng 0.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


class Status(str, Enum):
    """Kế thừa str để in ra, so sánh và đưa vào JSON đều tự nhiên."""

    SILENT = "SILENT"        # không có tín hiệu, và không có gì gần đây để hiện
    UNSTABLE = "UNSTABLE"    # có tín hiệu nhưng không tin được (nhiễu, đa âm, transient)
    ACQUIRING = "ACQUIRING"  # đang hiện số đọc nhưng chưa ổn định — chưa cho sáng "đúng"
    LOCKED = "LOCKED"        # số đọc ổn định, tin được
    HOLDING = "HOLDING"      # đang hiện số đọc CŨ; tín hiệu vừa mất. UI nên làm mờ.


class Direction(str, Enum):
    FLAT = "FLAT"            # thấp hơn target -> siết dây
    IN_TUNE = "IN_TUNE"
    SHARP = "SHARP"          # cao hơn target -> nới dây


#: status nào thì các trường đo được có giá trị
SHOWING = frozenset({Status.ACQUIRING, Status.LOCKED, Status.HOLDING})


@dataclass(frozen=True)
class TuningResult:
    """Hình dạng CỐ ĐỊNH. Trường vắng mặt là None, không phải biến mất."""

    # --- luôn có, mọi status ---
    status: Status
    confidence: float          # 0..1
    level_dbfs: float          # mức tín hiệu, cho gợi ý "chơi to hơn"
    timestamp_ms: float        # vị trí cuối frame trong luồng, đồng hồ đơn điệu

    # --- None trừ khi status ∈ SHOWING ---
    frequency_hz: Optional[float] = None       # ĐÃ làm mượt — số để hiển thị
    raw_frequency_hz: Optional[float] = None   # CHƯA làm mượt — để chẩn đoán
    midi_note: Optional[int] = None
    note_name: Optional[str] = None            # "A"  — tách khỏi octave
    octave: Optional[int] = None               # 4
    target_hz: Optional[float] = None
    cents: Optional[float] = None
    direction: Optional[Direction] = None

    #: số đọc đang hiện đã cũ bao nhiêu ms. 0 khi đang thật sự đo.
    stale_ms: float = 0.0

    def __post_init__(self):
        has = self.frequency_hz is not None
        if has != (self.status in SHOWING):
            raise ValueError(
                f"vi phạm hợp đồng: status={self.status} nhưng "
                f"frequency_hz={'có' if has else 'None'}")

    def as_dict(self):
        """Cho UI và cho ghi log JSONL. Enum thành str nhờ kế thừa str."""
        return asdict(self)
