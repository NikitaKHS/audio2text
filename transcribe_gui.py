"""GUI. Транскрибация в фоновом потоке, см. README."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from a2t_lib.config import COMPUTE_TYPES, DEVICES, MODELS, PRESETS, TranscribeConfig, TranscribeParams
from a2t_lib.engine import transcribe_file
from a2t_lib.gui_widgets import (
    COLORS,
    FONTS,
    RADIUS,
    BigFilePicker,
    Collapsible,
    FieldCombo,
    FieldEntry,
    OutputFolderRow,
    PresetCard,
    PrimaryButton,
    SecondaryButton,
    Section,
    StepHeader,
    ToggleRow,
    apply_theme,
)

PRESET_UI: dict[str, dict[str, str]] = {
    "safe": {
        "title": "Надёжный",
        "description": "Лучший выбор для длинных записей и интервью",
        "meta": "large-v3 · ~6 ГБ VRAM · рекомендуется",
    },
    "balanced": {
        "title": "Баланс",
        "description": "Компромисс между скоростью и точностью",
        "meta": "large-v3 · чуть быстрее надёжного",
    },
    "max_quality": {
        "title": "Максимум",
        "description": "Наивысшая точность, долго и требовательно к GPU",
        "meta": "float32 · ~10+ ГБ VRAM · медленно",
    },
    "fast": {
        "title": "Быстро",
        "description": "Черновик или короткие клипы — не для важных записей",
        "meta": "distil-large-v3 · минимум VRAM",
    },
}


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class TranscribeApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        apply_theme()

        self.title("audio2text")
        self.minsize(1120, 820)
        self.geometry("1280x900")
        self.configure(fg_color=COLORS["bg"])

        self._cancel = False
        self._worker: threading.Thread | None = None
        # GUI thread читает очередь; worker только put()
        self._msg_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._duration = 1.0
        self._preset_cards: dict[str, PresetCard] = {}

        # State
        self.audio_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.preset_var = tk.StringVar(value="safe")
        self.model_var = tk.StringVar(value="large-v3")
        self.device_var = tk.StringVar(value="cuda")
        self.compute_var = tk.StringVar(value="default")
        self.language_var = tk.StringVar(value="ru")
        self.beam_var = tk.StringVar(value="5")
        self.best_of_var = tk.StringVar(value="1")
        self.no_speech_var = tk.StringVar(value="0.75")
        self.condition_var = tk.BooleanVar(value=False)
        self.vad_var = tk.BooleanVar(value=True)
        self.postprocess_var = tk.BooleanVar(value=True)
        self.save_srt_var = tk.BooleanVar(value=True)
        self.save_plain_var = tk.BooleanVar(value=True)
        self.save_raw_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Готов к работе")
        self.progress_pct_var = tk.StringVar(value="0%")
        self.time_var = tk.StringVar(value="—")
        self.preview_var = tk.StringVar(value="Выберите файл и нажмите «Старт»")

        self._build_ui()
        self._select_preset("safe")
        self.after(100, self._poll_queue)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=24, pady=20)
        root.grid_columnconfigure(0, weight=3, minsize=520)
        root.grid_columnconfigure(1, weight=2, minsize=380)
        root.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(
            root,
            fg_color="transparent",
            scrollbar_button_color=COLORS["elevated"],
            scrollbar_button_hover_color=COLORS["border"],
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        right = ctk.CTkFrame(
            root,
            fg_color=COLORS["panel"],
            corner_radius=RADIUS["xl"],
            border_width=1,
            border_color=COLORS["border"],
        )
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_left(left)
        self._build_right(right)

        # Bottom action bar (full width)
        bar = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=0, height=80)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        bar_inner = ctk.CTkFrame(bar, fg_color="transparent")
        bar_inner.pack(fill="both", expand=True, padx=24, pady=14)

        self.start_btn = PrimaryButton(
            bar_inner,
            text="  Начать транскрибацию  ",
            width=280,
            command=self._start,
        )
        self.start_btn.pack(side="left")

        self.cancel_btn = SecondaryButton(
            bar_inner, text="Отмена", width=120,
            command=self._cancel_run, state="disabled",
        )
        self.cancel_btn.pack(side="left", padx=(12, 0))

        ctk.CTkLabel(
            bar_inner, textvariable=self.status_var,
            font=FONTS["label"], text_color=COLORS["text_muted"],
        ).pack(side="right")

    def _build_left(self, parent) -> None:
        ctk.CTkLabel(
            parent, text="audio2text",
            font=FONTS["hero"], text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            parent,
            text="Превратите аудио в текст с таймкодами",
            font=FONTS["body"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 24))

        # ── Шаг 1 ──
        StepHeader(
            parent, 1, "Выберите файл",
            "Укажите запись, которую нужно расшифровать",
        ).pack(fill="x", pady=(0, 10))

        sec1 = Section(parent)
        sec1.pack(fill="x", pady=(0, 28))
        BigFilePicker(
            sec1.body, self.audio_var, self._pick_audio,
        ).pack(fill="x")
        OutputFolderRow(
            sec1.body, self.output_var, self._pick_output,
        ).pack(fill="x", pady=(8, 0))

        # ── Шаг 2 ──
        StepHeader(
            parent, 2, "Режим качества",
            "Для русских длинных записей выбирайте «Надёжный»",
        ).pack(fill="x", pady=(0, 10))

        sec2 = Section(parent)
        sec2.pack(fill="x", pady=(0, 28))

        grid = ctk.CTkFrame(sec2.body, fg_color="transparent")
        grid.pack(fill="x")
        grid.grid_columnconfigure((0, 1), weight=1, uniform="p")

        for i, pid in enumerate(("safe", "balanced", "max_quality", "fast")):
            ui = PRESET_UI[pid]
            card = PresetCard(
                grid,
                preset_id=pid,
                title=ui["title"],
                description=ui["description"],
                meta=ui["meta"],
                on_select=self._select_preset,
            )
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            self._preset_cards[pid] = card

        # ── Шаг 3 ──
        StepHeader(
            parent, 3, "Что сохранить",
            "Рекомендуем оставить все пункты включёнными",
        ).pack(fill="x", pady=(0, 10))

        sec3 = Section(parent)
        sec3.pack(fill="x", pady=(0, 28))

        toggles = [
            ("Очистка текста", "Убирает повторы и мусор Whisper", self.postprocess_var),
            ("Субтитры SRT", "Файл с таймкодами для видеоплеера", self.save_srt_var),
            ("Текст без меток", "Сплошной текст для чтения", self.save_plain_var),
            ("Черновик raw", "До очистки — для отладки", self.save_raw_var),
        ]
        for title, sub, var in toggles:
            ToggleRow(sec3.body, title, sub, var).pack(fill="x", pady=5)

        # ── Расширенные ──
        adv = Collapsible(parent, "Расширенные настройки (модель, Whisper, контекст)")
        adv.pack(fill="x", pady=(0, 16))

        row_m = ctk.CTkFrame(adv.body, fg_color="transparent")
        row_m.pack(fill="x", pady=(0, 12))
        row_m.grid_columnconfigure((0, 1), weight=1)

        FieldCombo(
            row_m, "Модель", "Whisper-модель", self.model_var, MODELS, readonly=False,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        FieldCombo(
            row_m, "Устройство", "GPU быстрее", self.device_var, DEVICES,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row_m2 = ctk.CTkFrame(adv.body, fg_color="transparent")
        row_m2.pack(fill="x", pady=(0, 12))
        row_m2.grid_columnconfigure((0, 1), weight=1)
        FieldCombo(
            row_m2, "Точность VRAM", "float16 — стандарт", self.compute_var, COMPUTE_TYPES,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        FieldCombo(
            row_m2, "Язык", "ru для русского", self.language_var,
            ["ru", "en", "uk", "de", "fr"], readonly=False,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row_p = ctk.CTkFrame(adv.body, fg_color="transparent")
        row_p.pack(fill="x", pady=(0, 12))
        row_p.grid_columnconfigure((0, 1, 2), weight=1)
        FieldEntry(
            row_p, "Глубина поиска", "beam_size · 5–10 для качества", self.beam_var,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldEntry(
            row_p, "Кандидатов", "best_of", self.best_of_var,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        FieldEntry(
            row_p, "Порог тишины", "0.7–0.8 для длинных записей", self.no_speech_var,
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        ToggleRow(
            adv.body,
            "Учитывать предыдущий текст",
            "Может улучшить связность, но риск зацикливания на паузах",
            self.condition_var,
        ).pack(fill="x", pady=5)
        ToggleRow(
            adv.body,
            "Отсечение тишины (VAD)",
            "Пропускает длинные паузы без речи",
            self.vad_var,
        ).pack(fill="x", pady=5)

        ctk.CTkLabel(
            adv.body, text="Контекст для модели",
            font=FONTS["label"], text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w", pady=(16, 4))
        ctk.CTkLabel(
            adv.body,
            text="Опишите тему разговора — имена, термины, обстановку",
            font=FONTS["caption"], text_color=COLORS["text_muted"], anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        self.prompt_entry = ctk.CTkTextbox(
            adv.body, height=100, font=FONTS["body"],
            fg_color=COLORS["input"], border_color=COLORS["border"],
            border_width=1, corner_radius=RADIUS["sm"],
            text_color=COLORS["text"],
        )
        self.prompt_entry.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            adv.body, text="Ключевые слова",
            font=FONTS["label"], text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w", pady=(0, 4))
        self.hotwords_entry = ctk.CTkTextbox(
            adv.body, height=64, font=FONTS["body"],
            fg_color=COLORS["input"], border_color=COLORS["border"],
            border_width=1, corner_radius=RADIUS["sm"],
            text_color=COLORS["text"],
        )
        self.hotwords_entry.pack(fill="x")

    def _build_right(self, parent) -> None:
        pad = ctk.CTkFrame(parent, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=22, pady=22)

        ctk.CTkLabel(
            pad, text="Прогресс",
            font=FONTS["h1"], text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w")

        stat_row = ctk.CTkFrame(pad, fg_color="transparent")
        stat_row.pack(fill="x", pady=(20, 8))
        ctk.CTkLabel(
            stat_row, textvariable=self.progress_pct_var,
            font=FONTS["stat"], text_color=COLORS["accent_hover"],
        ).pack(side="left")
        ctk.CTkLabel(
            stat_row, textvariable=self.time_var,
            font=FONTS["h2"], text_color=COLORS["text_muted"],
        ).pack(side="right", pady=(8, 0))

        self.progress = ctk.CTkProgressBar(
            pad, height=10, corner_radius=5,
            fg_color=COLORS["elevated"],
            progress_color=COLORS["accent"],
        )
        self.progress.pack(fill="x", pady=(0, 16))
        self.progress.set(0)

        preview_box = ctk.CTkFrame(
            pad,
            fg_color=COLORS["input"],
            corner_radius=RADIUS["md"],
            border_width=1,
            border_color=COLORS["border"],
        )
        preview_box.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(
            preview_box,
            text="Сейчас распознаётся",
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(
            preview_box,
            textvariable=self.preview_var,
            font=FONTS["body"],
            text_color=COLORS["text"],
            anchor="w",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(6, 16))

        ctk.CTkLabel(
            pad, text="Журнал",
            font=FONTS["h2"], text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        self.log = ctk.CTkTextbox(
            pad,
            font=FONTS["mono"],
            fg_color=COLORS["log"],
            text_color=COLORS["text_secondary"],
            corner_radius=RADIUS["md"],
            border_width=1,
            border_color=COLORS["border"],
            wrap="word",
            activate_scrollbars=True,
        )
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

    # ── Logic ────────────────────────────────────────────────────────────────

    def _select_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        self.preset_var.set(name)
        meta = PRESETS[name]
        self.model_var.set(meta["model"])
        self.compute_var.set(meta["compute_type"])
        p: TranscribeParams = meta["params"]
        self.beam_var.set(str(p.beam_size))
        self.best_of_var.set(str(p.best_of))
        self.no_speech_var.set(str(p.no_speech_threshold))
        self.condition_var.set(p.condition_on_previous_text)
        self.vad_var.set(p.vad_filter)
        for pid, card in self._preset_cards.items():
            card.set_selected(pid == name)
        if not self.prompt_entry.get("1.0", "end").strip():
            from a2t_lib.config import DEFAULT_HOTWORDS, DEFAULT_PROMPT
            self.prompt_entry.delete("1.0", "end")
            self.prompt_entry.insert("1.0", DEFAULT_PROMPT)
            self.hotwords_entry.delete("1.0", "end")
            self.hotwords_entry.insert("1.0", DEFAULT_HOTWORDS)

    def _on_preset_change(self) -> None:
        self._select_preset(self.preset_var.get())

    def _pick_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудио", "*.m4a *.mp3 *.wav *.ogg *.flac *.webm *.mp4 *.mkv"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self.audio_var.set(path)
            if not self.output_var.get():
                self.output_var.set(str(Path(path).parent))
            self.preview_var.set(f"Файл: {Path(path).name}")
            self.status_var.set("Файл выбран — можно начинать")

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Папка для результатов")
        if path:
            self.output_var.set(path)

    def _append_log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _build_config(self) -> TranscribeConfig:
        audio = Path(self.audio_var.get().strip())
        out = self.output_var.get().strip()
        cfg = TranscribeConfig(
            audio_path=audio,
            output_dir=Path(out) if out else None,
            model=self.model_var.get(),
            device=self.device_var.get(),
            compute_type=self.compute_var.get(),
            language=self.language_var.get(),
            preset=self.preset_var.get(),
            initial_prompt=self.prompt_entry.get("1.0", "end").strip(),
            hotwords=self.hotwords_entry.get("1.0", "end").strip(),
            postprocess=self.postprocess_var.get(),
            save_srt=self.save_srt_var.get(),
            save_plain=self.save_plain_var.get(),
            save_raw=self.save_raw_var.get(),
        )
        cfg.params = TranscribeParams(
            beam_size=int(float(self.beam_var.get())),
            best_of=int(float(self.best_of_var.get())),
            no_speech_threshold=float(self.no_speech_var.get()),
            condition_on_previous_text=self.condition_var.get(),
            vad_filter=self.vad_var.get(),
        )
        if cfg.preset in PRESETS:
            base = PRESETS[cfg.preset]["params"]
            for field in (
                "patience", "temperature", "vad_threshold", "vad_min_speech_ms",
                "vad_min_silence_ms", "word_timestamps", "compression_ratio_threshold",
                "hallucination_silence_threshold", "low_confidence_logprob",
            ):
                setattr(cfg.params, field, getattr(base, field))
        return cfg

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        audio = self.audio_var.get().strip()
        if not audio:
            messagebox.showwarning("audio2text", "Сначала выберите аудиофайл")
            return
        if not Path(audio).is_file():
            messagebox.showerror("audio2text", f"Файл не найден:\n{audio}")
            return

        self._cancel = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.progress_pct_var.set("0%")
        self.time_var.set("—")
        self.status_var.set("Запуск…")

        cfg = self._build_config()
        self._append_log(f"[Старт] {cfg.audio_path.name} | {PRESET_UI[cfg.preset]['title']}")

        def run() -> None:
            try:
                transcribe_file(
                    cfg,
                    log=lambda m: self._msg_queue.put(("log", m)),
                    progress=lambda s, sn: self._msg_queue.put(("progress", (s, sn))),
                    cancel_check=lambda: self._cancel,
                )
                self._msg_queue.put(("done", None))
            except InterruptedError:
                self._msg_queue.put(("cancelled", None))
            except Exception as e:
                self._msg_queue.put(("error", str(e)))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _cancel_run(self) -> None:
        self._cancel = True
        self.status_var.set("Отмена…")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    sec, snippet = payload  # type: ignore[misc]
                    if sec < 0:
                        self._duration = -sec
                        self.progress.set(0)
                        self.time_var.set(f"0:00 / {_fmt_time(self._duration)}")
                    else:
                        pct = min(1.0, sec / self._duration) if self._duration else 0
                        self.progress.set(pct)
                        self.progress_pct_var.set(f"{int(pct * 100)}%")
                        self.time_var.set(f"{_fmt_time(sec)} / {_fmt_time(self._duration)}")
                        if snippet:
                            self.preview_var.set(snippet)
                            self.status_var.set("Идёт транскрибация…")
                elif kind == "done":
                    self._finish(success=True)
                elif kind == "cancelled":
                    self._finish(success=False, msg="Отменено")
                elif kind == "error":
                    self._finish(success=False, msg="Ошибка")
                    messagebox.showerror("Ошибка", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish(self, success: bool, msg: str = "Готово") -> None:
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.status_var.set(msg)
        if success:
            self.progress.set(1.0)
            self.progress_pct_var.set("100%")
            self.preview_var.set("Транскрибация завершена")
            self._append_log("[Готово] Все файлы сохранены")


# Re-export for tests
PRESET_LABELS = {k: v["title"] for k, v in PRESET_UI.items()}
LABEL_TO_PRESET = {v["title"]: k for k, v in PRESET_UI.items()}


def main() -> None:
    app = TranscribeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
