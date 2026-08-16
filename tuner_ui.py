"""
Instrument Tuner — UI with tkinter
==============================
Cơ chế: Always-on + Auto-freeze
  - Mic luôn chạy ngầm
  - Khi phát hiện âm → cập nhật display
  - Khi im lặng > FREEZE_DELAY giây → màn hình freeze, giữ kết quả cuối

Chạy project:
    python tuner_ui.py
"""

import tkinter as tk
import sys
import time

from tuner.io import MicSource, TunerRunner

# ── Khởi tạo engine ──────────────────────────────────────────
_runner = TunerRunner(MicSource())
_runner.start()
# ─────────────────────────────────────────────────────────────


# ── Màu sắc (dark theme) ──────────────────────────────────────
BG          = "#111111"
SURFACE     = "#1c1c1e"
SURFACE2    = "#2c2c2e"
BORDER      = "#3a3a3c"
TEXT_PRI    = "#f5f5f7"
TEXT_SEC    = "#8e8e93"
TEXT_MUT    = "#48484a"

FLAT_C      = "#ff453a"   # đỏ — mỏng hơn (flat)
FLAT_BG     = "#3a1816"
SHARP_C     = "#ffd60a"   # vàng — dày hơn (sharp)
SHARP_BG    = "#332d00"
INTUNE_C    = "#30d158"   # xanh — chuẩn
INTUNE_BG   = "#0d2416"
NEUTRAL_C   = "#636366"

# ── Font ──────────────────────────────────────────────────────
if sys.platform == "darwin":
    F = "SF Pro Display"
elif sys.platform == "win32":
    F = "Segoe UI"
else:
    F = "DejaVu Sans"


# ─────────────────────────────────────────────────────────────
class TunerApp:
    FREEZE_DELAY = 1.5   # giây im lặng trước khi freeze
    POLL_MS      = 50    # polling interval (20 fps)
    MAX_CENTS    = 50    # ±50¢ là biên của thanh bar
    DOT_R        = 11    # bán kính chấm tròn trên bar
    TRACK_H      = 8     # chiều cao thanh bar

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Instrument Tuner")
        self.root.geometry("580x490")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Trạng thái hiện tại
        self._note       = "—"
        self._cents      = 0.0
        self._hz         = 0.0
        self._target_hz  = 0.0
        self._frozen     = True
        self._last_signal_time = 0.0
        self._blink_on   = True

        self._build_ui()
        self._poll()

        # Dừng engine khi đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        _runner.stop()
        self.root.destroy()

    # ──────────────────────────────────────────────────────────
    # BUILD UI
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", pady=(20, 0))
        tk.Label(hdr, text="INSTRUMENT TUNER",
                 font=(F, 10), fg=TEXT_MUT, bg=BG).pack()

        # Divider
        tk.Frame(self.root, bg=BORDER, height=1).pack(
            fill="x", padx=40, pady=14)

        # ── Note display ──
        note_frame = tk.Frame(self.root, bg=BG)
        note_frame.pack()

        self.lbl_note = tk.Label(
            note_frame, text="—",
            font=(F, 80, "bold"),
            fg=TEXT_PRI, bg=BG, width=4, anchor="center")
        self.lbl_note.pack()

        self.lbl_hz = tk.Label(
            note_frame,
            text="gảy đàn để bắt đầu",
            font=(F, 12), fg=TEXT_MUT, bg=BG)
        self.lbl_hz.pack(pady=(0, 4))

        # ── Pitch bar ──
        bar_container = tk.Frame(self.root, bg=BG)
        bar_container.pack(fill="x", padx=48, pady=(18, 4))

        self.bar_canvas = tk.Canvas(
            bar_container, height=60,
            bg=BG, highlightthickness=0)
        self.bar_canvas.pack(fill="x")
        self.bar_canvas.bind("<Configure>", self._on_bar_resize)

        # Edge labels
        edge = tk.Frame(self.root, bg=BG)
        edge.pack(fill="x", padx=48)
        tk.Label(edge, text="← mỏng hơn  (flat)",
                 font=(F, 10), fg=TEXT_MUT, bg=BG).pack(side="left")
        tk.Label(edge, text="(sharp)  dày hơn →",
                 font=(F, 10), fg=TEXT_MUT, bg=BG).pack(side="right")

        # ── Status pill ──
        pill_wrap = tk.Frame(self.root, bg=BG)
        pill_wrap.pack(pady=22)

        self.pill_frame = tk.Frame(
            pill_wrap,
            bg=SURFACE2,
            highlightbackground=BORDER,
            highlightthickness=1)
        self.pill_frame.pack()

        self.lbl_status = tk.Label(
            self.pill_frame,
            text="Gảy đàn để bắt đầu",
            font=(F, 13), fg=TEXT_SEC, bg=SURFACE2,
            padx=28, pady=11)
        self.lbl_status.pack()

        # ── Listening indicator ──
        listen_row = tk.Frame(self.root, bg=BG)
        listen_row.pack()

        self.dot_canvas = tk.Canvas(
            listen_row, width=10, height=10,
            bg=BG, highlightthickness=0)
        self.dot_canvas.pack(side="left", padx=(0, 8))

        self.lbl_listen = tk.Label(
            listen_row, text="Đang nghe...",
            font=(F, 11), fg=TEXT_SEC, bg=BG)
        self.lbl_listen.pack(side="left")

        self.lbl_frozen = tk.Label(
            listen_row, text="",
            font=(F, 10), fg=TEXT_MUT, bg=BG)
        self.lbl_frozen.pack(side="left", padx=(12, 0))

        # Canvas item IDs (populated in _on_bar_resize)
        self._bar_ready  = False
        self._zone_id    = None
        self._dot_id     = None
        self._center_lbl = None
        self._bar_w      = 0

    # ──────────────────────────────────────────────────────────
    # BAR CANVAS
    # ──────────────────────────────────────────────────────────
    def _on_bar_resize(self, event):
        self._draw_bar_base(event.width)

    def _draw_bar_base(self, w):
        c = self.bar_canvas
        c.delete("all")
        if w < 20:
            return

        self._bar_w = w
        mid = w // 2
        ty  = 30   # vertical center of bar

        # Track background
        c.create_rectangle(
            0, ty - self.TRACK_H // 2,
            w, ty + self.TRACK_H // 2,
            fill=SURFACE2, outline=BORDER, width=1,
            tags="track")

        # Colored zone (starts empty)
        self._zone_id = c.create_rectangle(
            0, 0, 0, 0, fill="", outline="", tags="zone")

        # Center tick
        c.create_rectangle(
            mid - 1, ty - 18, mid + 1, ty + 18,
            fill=BORDER, outline="", tags="center_tick")

        # Center label (note name above tick)
        self._center_lbl = c.create_text(
            mid, ty - 26, text="",
            font=(F, 10), fill=TEXT_SEC,
            tags="center_label")

        # Dot (drawn on top)
        r = self.DOT_R
        self._dot_id = c.create_oval(
            mid - r, ty - r, mid + r, ty + r,
            fill=SURFACE, outline=BORDER, width=2,
            tags="dot")

        # zone stays below dot
        c.tag_lower("zone", "dot")

        self._bar_ready = True
        self._repaint_bar()

    def _repaint_bar(self):
        if not self._bar_ready:
            return

        c   = self.bar_canvas
        w   = self._bar_w
        mid = w // 2
        ty  = 30
        r   = self.DOT_R

        # Dot X position
        if self._note == "—":
            x = mid
            dot_fill    = SURFACE
            dot_outline = BORDER
            zone_coords = (0, 0, 0, 0)
            zone_color  = ""
        else:
            clamped = max(-self.MAX_CENTS,
                          min(self.MAX_CENTS, self._cents))
            travel  = (w / 2) - r - 4
            x = mid + (clamped / self.MAX_CENTS) * travel

            if abs(self._cents) <= 5:
                dot_fill    = INTUNE_BG
                dot_outline = INTUNE_C
                zone_coords = (0, 0, 0, 0)
                zone_color  = ""
            elif self._cents < 0:
                dot_fill    = FLAT_BG
                dot_outline = FLAT_C
                zone_coords = (x, ty - self.TRACK_H // 2,
                               mid, ty + self.TRACK_H // 2)
                zone_color  = FLAT_BG
            else:
                dot_fill    = SHARP_BG
                dot_outline = SHARP_C
                zone_coords = (mid, ty - self.TRACK_H // 2,
                               x, ty + self.TRACK_H // 2)
                zone_color  = SHARP_BG

        # Apply
        c.coords(self._dot_id,
                 x - r, ty - r, x + r, ty + r)
        c.itemconfig(self._dot_id,
                     fill=dot_fill, outline=dot_outline)

        c.coords(self._zone_id, *zone_coords)
        c.itemconfig(self._zone_id, fill=zone_color, outline="")

        # Center label
        lbl = (f"{self._note}  ({self._target_hz:.1f} Hz)"
               if self._note != "—" else "")
        c.itemconfig(self._center_lbl, text=lbl)

    # ──────────────────────────────────────────────────────────
    # DISPLAY UPDATE
    # ──────────────────────────────────────────────────────────
    def _update_display(self):
        """Cập nhật toàn bộ UI theo trạng thái hiện tại."""
        note  = self._note
        cents = self._cents

        self.lbl_note.config(text=note)

        if self._hz > 0:
            self.lbl_hz.config(
                text=f"Phát hiện: {self._hz:.1f} Hz",
                fg=TEXT_SEC)

        # Status pill
        if note == "—":
            txt    = "Gảy đàn để bắt đầu"
            fg     = TEXT_SEC
            bg     = SURFACE2
            border = BORDER
        elif abs(cents) <= 5:
            txt    = "Chuẩn rồi"
            fg     = INTUNE_C
            bg     = INTUNE_BG
            border = INTUNE_C
        elif cents < 0:
            txt    = f"Mỏng hơn {abs(cents):.0f}¢  —  kéo căng thêm"
            fg     = FLAT_C
            bg     = FLAT_BG
            border = FLAT_C
        else:
            txt    = f"Dày hơn {cents:.0f}¢  —  nới lỏng ra"
            fg     = SHARP_C
            bg     = SHARP_BG
            border = SHARP_C

        self.lbl_status.config(text=txt, fg=fg, bg=bg)
        self.pill_frame.config(bg=bg, highlightbackground=border)

        self._repaint_bar()

    def _set_frozen(self):
        """Gọi khi signal mất đủ lâu → freeze màn hình."""
        self.lbl_hz.config(
            text="— gảy lại để cập nhật —",
            fg=TEXT_MUT)
        self.lbl_frozen.config(
            text="[đã dừng]", fg=TEXT_MUT)

    def _set_listening(self):
        """Gọi khi nhận được signal trở lại."""
        self.lbl_frozen.config(text="")

    # ──────────────────────────────────────────────────────────
    # LISTENING DOT BLINK
    # ──────────────────────────────────────────────────────────
    def _blink_dot(self):
        c = self.dot_canvas
        c.delete("all")
        color = INTUNE_C if self._blink_on else TEXT_MUT
        c.create_oval(1, 1, 9, 9, fill=color, outline="")
        self._blink_on = not self._blink_on

    # ──────────────────────────────────────────────────────────
    # POLL LOOP  (chạy mỗi POLL_MS ms)
    # ──────────────────────────────────────────────────────────
    def _poll(self):
        _runner.poll()
        r   = _runner.latest
        now = time.time()

        # Chỉ hiện khi engine đã khoá được nốt (ACQUIRING hoặc LOCKED)
        if r is not None and r.status.name in ("ACQUIRING", "LOCKED"):
            self._note      = r.note_name + str(r.octave)
            self._cents     = r.cents
            self._hz        = r.frequency_hz
            self._target_hz = r.target_hz
            self._last_signal_time = now

            if self._frozen:
                self._frozen = False
                self._set_listening()

            self._update_display()

        else:
            elapsed = now - self._last_signal_time
            if not self._frozen and elapsed > self.FREEZE_DELAY:
                self._frozen = True
                self._set_frozen()

        self._blink_dot()
        self.root.after(self.POLL_MS, self._poll)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = TunerApp(root)
    root.mainloop()
