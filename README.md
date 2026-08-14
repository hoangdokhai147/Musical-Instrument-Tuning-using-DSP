# Musical Instrument Tuner — DSP Engine

Bộ máy nhận diện cao độ thời gian thực cho nhạc cụ đơn âm. Toàn bộ phần xử lý tín
hiệu **tự cài từ đầu**, chỉ dùng `numpy` cho thao tác mảng.

Đo được qua trọn chuỗi âm học (loa → không khí → mic → ADC → engine): sai số
**−0.26 cent**. Xử lý **2.9 ms** cho mỗi frame trên nhịp cập nhật 21.3 ms, tức 14%
một core.

Đồ án môn DSP501 — Xử lý tín hiệu số.

---

## Chạy thử

```bash
pip install -r requirements.txt

python console_check.py                     # dùng microphone
python console_check.py --list              # liệt kê thiết bị vào
python console_check.py --file recordings/hard/guitar_notechange.wav --loop
```

Màn hình:

```
  LOCKED    A2   109.98 Hz →  110.00  ─────────┼●────────  -0.3¢ IN_TUNE conf 0.99
  HOLDING   A2   109.98 Hz →  110.00  ─────────┼●────────  -0.3¢ IN_TUNE (cũ 340 ms)
  SILENT    chơi một nốt
```

> **macOS:** lần đầu chạy sẽ cần cấp quyền micro. macOS cấp quyền cho *ứng dụng
> terminal* chứ không cho Python, và khi chưa cấp thì nó **trả về toàn số 0 mà
> không báo lỗi**. Engine phát hiện việc này trong ~1 giây đầu và in hướng dẫn.
> Cũng cần kiểm tra Control Center → Micrô đang ở chế độ **"Chuẩn"**, không phải
> "Cách ly giọng nói".

---

## Kết quả đo được

Tất cả đều là số đo trên máy, không phải ước lượng. Bộ dữ liệu kiểm thử nằm trong
`recordings/` (32 file, guitar thép + tone hiệu chuẩn + đối chứng số).

| Phép đo | Kết quả | Đo trên |
|---|---|---|
| Sai số tuyệt đối, chuỗi âm học đầy đủ | **−0.26 cent** | tone chuẩn qua loa + mic |
| Sai số, tín hiệu số thuần | 0.078 cent | 7 file đối chứng |
| Sai số, sweep tổng hợp 65–1100 Hz | max 0.45 cent | 2070 điểm |
| Jitter (cửa sổ 1 giây) | 0.37 – 5.81 cent | 6 dây guitar thật |
| Lỗi octave | 0.28 % | 6 dây guitar thật |
| Thời gian tới số đọc đầu tiên | 114 ms | hop 1024 @ 48 kHz |
| Khoảng cách hai lần cập nhật | 21.3 ms | 47 lần/giây |
| Chi phí xử lý | 2.9 ms/frame | 14% một core |
| Dải nhận được | 65 – 1100 Hz | C2 … ~C#6 |

Con số **−0.26 cent** là quan trọng nhất: nó đo qua *toàn bộ* chuỗi, nên mọi độ
lệch quan sát được trên nhạc cụ thật đều quy được cho cây đàn chứ không phải cho
thuật toán. Đó là lý do bộ thu có riêng nhóm tone hiệu chuẩn.

---

## Thuật toán

**YIN** (de Cheveigné & Kawahara, 2002) — ước lượng cao độ trong miền thời gian.

1. `d(τ) = Σ (x[j] − x[j+τ])²` — hàm sai khác, viết thẳng theo định nghĩa
2. `d'(τ) = d(τ)·τ / Σ_{j≤τ} d(j)` — chuẩn hoá trung bình tích luỹ (CMNDF)
3. Lấy τ **nhỏ nhất** có `d'(τ)` dưới ngưỡng, rồi trượt xuống đáy cục bộ
4. Nội suy parabol để lấy τ dưới mức mẫu
5. `f0 = fs / τ*`

Cộng một lớp canh gác **không thuộc YIN**: kiểm chứng phổ bằng Goertzel. Vì
`d(nT) ≈ 0` với *mọi* bội n, YIN có thể trả về một chu kỳ đúng về mặt toán học mà
vô nghĩa về âm nhạc. Confidence không bắt được — một subharmonic vẫn thực sự tuần
hoàn. Chỉ miền tần số mới phân biệt được.

### Bốn quyết định thiết kế đáng nói

**Không hạ mẫu.** Giữ nguyên 48 kHz cho độ phân giải lag mịn gấp 4 (5.4 cent/mẫu
ở 220 Hz thay vì 21.7), nên nội suy dưới mức mẫu phải gánh ít hơn.

**Không dùng FFT trong đường pitch.** `d(τ)` tính thẳng theo định nghĩa nhanh hơn
bản FFT tự cài **18–79 lần** — lợi thế O(N log N) bị hằng số ~100:1 nuốt trọn, vì
FFT chạy trong vòng lặp Python còn cách trực tiếp chạy trong vòng lặp C của numpy.
FFT vẫn được giữ, nhưng ở đúng chỗ của nó: phổ hiển thị (`tuner/analysis/`).

**Không cửa sổ hoá trước YIN.** `d(τ)` so sánh `x[j]` với `x[j+τ]`; nhân cửa sổ
vào khiến hai mẫu được so chịu hai trọng số khác nhau, tạo sai khác giả tăng theo
τ — tức thiên vị lag ngắn, tức gây lỗi octave. Cửa sổ hoá ở đây không chỉ thừa,
nó chủ động có hại. (Với FFT hiển thị thì ngược lại: Hann là bắt buộc.)

**Nội suy trên `d(τ)`, không phải `d'(τ)`.** `d'` là `d` nhân một hàm tăng theo τ,
làm nghiêng parabol và đẩy đỉnh sang trái. Đo được: bias hệ thống +0.28 cent trung
bình, tăng tới +0.54 cent ở dải cao. Nội suy trên `d` đưa bias về −0.008 cent.

---

## Kiến trúc

```
tuner/
├── core/          DSP thuần. Không IO, không thread, không mili giây.
│   ├── config.py      Config — frozen dataclass, tự kiểm tra ràng buộc
│   ├── filters.py     thiết kế FIR + StreamingFIR      ← state của LUỒNG
│   ├── yin.py         bốn bước YIN + kiểm chứng subharmonic
│   ├── detector.py    PitchDetector                     ← KHÔNG state
│   ├── music.py       f0 ↔ MIDI ↔ tên nốt ↔ cent
│   └── tracking.py    NoteTracker                       ← state của NỐT
├── engine/
│   ├── framer.py      ring buffer + nhịp hop
│   ├── presenter.py   hysteresis, ACQUIRING, giữ số đọc ← state HIỂN THỊ
│   ├── result.py      TuningResult, Status, Direction
│   └── engine.py      Tuner — điều phối, tự nó không giữ state nào
├── io/            Adapter. core/ không bao giờ import từ đây.
│   ├── sources.py     MicSource, FileSource
│   └── runner.py      TunerRunner
└── analysis/      Công cụ dev. Không nằm trong đường tín hiệu.
    └── spectrum.py    FFT tự cài + Hann
```

Quy tắc phụ thuộc: **`core/` chỉ được import `numpy` và `core/`** — có test kiểm tra.

`core/yin.py` không biết gì về sample rate hay nốt nhạc, nó chỉ làm việc với *lag
tính bằng mẫu*. Ranh giới đó là lý do engine tự động đúng ở cả 44.1 và 48 kHz mà
không phải sửa dòng nào.

### Ba loại state, ba vòng đời

| Sở hữu bởi | State | Xoá khi |
|---|---|---|
| `StreamingFIR`, `Framer` | đuôi bộ lọc, cửa sổ đang tích luỹ | mở/đóng stream, đổi thiết bị |
| `NoteTracker` | median, EMA, bộ đếm frame hụt | người dùng thật sự ngừng chơi |
| `Presenter` | hysteresis, đếm ổn định, giữ số đọc | — (thuần hiển thị) |
| `PitchDetector` | **không có** | — |

Bản đầu tiên gộp hai loại đầu vào một `reset()`. Nghe như lỗi thẩm mỹ, thực ra là
nguyên nhân của **jitter 25 cent**: mỗi frame tụt dưới ngưỡng lại xoá luôn bộ đệm
median, nên `median=15` chưa bao giờ đầy nổi và hành xử y như `median=1`. Tách ra
đưa jitter về 0.37–5.81 cent.

---

## Dùng engine

```python
from tuner.io import MicSource, TunerRunner

runner = TunerRunner(MicSource())
runner.start()

# trong vòng lặp vẽ — tkinter: root.after(33, tick)
def tick():
    runner.poll()              # xử lý audio đã về
    r = runner.latest          # TuningResult, hoặc None nếu chưa có gì
    if r is not None:
        ve(r)
    root.after(33, tick)

runner.stop()                  # hoặc: with TunerRunner(...) as runner:
```

Ring buffer, kích thước chunk, bỏ audio quá cũ, khớp sample rate — đã lo bên trong.

Muốn tự quản luồng thì gọi thẳng:

```python
from tuner import Config, Tuner
tuner = Tuner(Config(fs=48000))
results = tuner.push(chunk)    # chunk dài BẤT KỲ → list[TuningResult]
```

### TuningResult

Một luật, mọi thứ suy ra từ nó:

> Các trường đo được khác `None` **khi và chỉ khi** `status` là `ACQUIRING`,
> `LOCKED` hoặc `HOLDING`.

Không có chuỗi rỗng nghĩa là "chưa có", không có `f0 = 0` nghĩa là im lặng. Engine
tự cưỡng chế — một result vi phạm không tạo ra được.

| status | nghĩa là | UI nên |
|---|---|---|
| `SILENT` | không có tín hiệu | "Chơi một nốt", xoá màn hình |
| `UNSTABLE` | có tiếng nhưng không ra cao độ | giữ khung, hiện "…". Không đoán bừa. |
| `ACQUIRING` | bắt được nốt, chưa ổn định | hiện nốt, kim mờ. Chưa sáng đèn "đúng". |
| `LOCKED` | ổn định, tin được | hiện đầy đủ |
| `HOLDING` | đang hiện số **cũ**, tín hiệu vừa mất | làm mờ theo `stale_ms`, tự hết sau 800 ms |

Trường: `status` `confidence` `level_dbfs` `timestamp_ms` `stale_ms` — luôn có.
`frequency_hz` `raw_frequency_hz` `note_name` `octave` `midi_note` `target_hz`
`cents` `direction` — chỉ khi đang hiện.

`note_name` và `octave` tách riêng vì UI render chúng ở hai cỡ chữ khác nhau.
`r.as_dict()` cho ra dict JSON-hoá được.

---

## Test

```bash
python tests/golden_vectors.py check   # hành vi có đổi không (~40 s)
python tests/test_config.py            #  8 test
python tests/test_state.py             # 10
python tests/test_framer.py            # 12
python tests/test_result.py            # 15
python tests/test_io.py                # 14
```

**Golden vectors** đóng băng hành vi hiện tại: 2070 vector tổng hợp + 32 file +
25 luồng đầy đủ. Sửa gì trong `tuner/` thì chạy `check` trước khi commit — nó chỉ
ra chính xác file/frame nào lệch. Nếu bạn *cố ý* đổi hành vi thì `freeze` lại và
ghi lý do vào commit; đừng freeze chỉ để test hết đỏ.

Bộ này cũng là **hợp đồng nghiệm thu cho bản port mobile**: khớp trong 0.01 cent.
Tín hiệu được mô tả bằng *spec* (seed, số hài, mức nhiễu) chứ không nhúng mảng mẫu
— để bản port phải tự sinh lại, và chính việc đó phát hiện sai lệch ULP trong
`sin`/`hanning`/`sinc` giữa hai ngôn ngữ.

> Bài học rút ra khi làm: cả ba bug thật tìm được trong quá trình audit đều **lọt
> qua nếu chỉ test ở các nốt chuẩn**. Chúng chỉ lộ ra khi quét f0 *liên tục*. Vì
> thế nhóm sweep là bắt buộc, và mọi test sweep khẳng định cả `|mean| < 0.05 cent`
> (bắt bias) chứ không chỉ `max|err|` (bắt biên độ).

---

## Giới hạn đã biết

Đây là hành vi đã đo và hiểu, không phải bug chưa sửa.

1. **f0 trên 1100 Hz bị báo thành subharmonic.** Ngoài dải thiết kế. Bồi âm ngăn
   12 dây 1 (659 Hz) vẫn đúng.
2. **Fundamental yếu hơn ~30% so với hài bậc 2** thì engine khoá vào hài âm.
3. **Dây G3 kém ổn định hơn** — jitter 5.8 cent so với 0.4–1.2 của năm dây kia.
   Nguyên nhân: fundamental của G3 yếu hơn hài bậc 4 tới 10 dB trên cây đàn đã thu.
   Đặc tính nhạc cụ, không phải lỗi engine.
4. **Đơn âm.** Hợp âm và hai dây cùng lúc bị từ chối (`UNSTABLE`), không đoán bừa.

---

## Bản đồ repo

```
tuner/                gói engine — xem phần Kiến trúc
tests/                59 test + golden vectors
recordings/           32 file kiểm thử (56 MB)
  calib/                tone hiệu chuẩn — ground truth tuyệt đối
  digital/              tone thuần số, không qua loa/mic — nhóm đối chứng
  steady/ transient/    guitar thép: dây buông, gảy rời
  hard/ env/ reject/    ca khó, môi trường ồn, ca phải bị từ chối
console_check.py      chạy engine với audio thật, in ra terminal
record_session.py     thu bộ dữ liệu kiểm thử
```

---

## Môi trường

Python 3.11 · numpy · sounddevice · soundfile

Đã kiểm chứng trên macOS (Darwin 25.4) với Python 3.11.15, numpy 2.4.6,
sounddevice 0.5.5, soundfile 0.14.0.
