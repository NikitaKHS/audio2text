"""GUI. Транскрибация в фоновом потоке, см. README."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from a2t_lib.accuracy import accuracy_installed, transcribe_high_accuracy
from a2t_lib.config import (
    COMPUTE_TYPES,
    DEVICES,
    MODELS,
    PRESETS,
    TranscribeConfig,
    TranscribeParams,
)
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
from a2t_lib.hardware import detect_gpus, hardware_summary, vram_warning

PRESET_UI: dict[str, dict[str, str]] = {
    "safe": {
        "title": "Надёжный",
        "description": "Лучший выбор для длинных записей и интервью",
        "meta": "large-v3 · float16 · рекомендуется для 8 ГБ VRAM",
    },
    "balanced": {
        "title": "Баланс",
        "description": "Компромисс между скоростью и точностью",
        "meta": "large-v3 · чуть быстрее надёжного",
    },
    "max_quality": {
        "title": "Максимум",
        "description": "Наивысшая точность, долго и требовательно к GPU",
        "meta": "float32 · требуется больше 10 ГБ VRAM · медленно",
    },
    "fast": {
        "title": "Быстро",
        "description": "Черновик или короткие клипы — не для важных записей",
        "meta": "distil-large-v3 · минимум VRAM",
    },
}

ENGINE_UI: dict[str, dict[str, str]] = {
    "whisper": {
        "title": "Whisper GPU",
        "description": "Быстрое распознавание любых поддерживаемых языков",
        "meta": "large-v3 · NVIDIA GPU или CPU",
        "badge": "БЫСТРО",
    },
    "gigaam": {
        "title": "GigaAM-v3 RNNT + CTC",
        "description": "Два декодера сверяют каждый фрагмент русской речи",
        "meta": "максимальная точность RU · полностью локально",
        "badge": "ТОЧНЕЕ RU",
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
        self.minsize(980, 560)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(1220, max(1000, int(screen_width * 0.78)))
        height = min(760, max(600, int(screen_height * 0.68)))
        x = max(0, (screen_width - width) // 2)
        y = 16
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.configure(fg_color=COLORS["bg"])

        self._cancel = False
        self._worker: threading.Thread | None = None
        # GUI thread читает очередь; worker только put()
        self._msg_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._duration = 1.0
        self._preset_cards: dict[str, PresetCard] = {}
        self._engine_cards: dict[str, PresetCard] = {}
        self._gpus = detect_gpus()

        # State
        self.audio_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.engine_var = tk.StringVar(value="whisper")
        self.engine_badge_var = tk.StringVar(value="WHISPER GPU")
        self.corrections_var = tk.StringVar()
        self.preset_var = tk.StringVar(value="safe")
        self.model_var = tk.StringVar(value="large-v3")
        self.device_var = tk.StringVar(value="auto")
        self.device_index_var = tk.StringVar(value="0")
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
        self.save_review_var = tk.BooleanVar(value=True)
        self.resume_var = tk.BooleanVar(value=True)
        self.hardware_var = tk.StringVar(value=hardware_summary(self._gpus))
        self.status_var = tk.StringVar(value="Готов к работе")
        self.progress_pct_var = tk.StringVar(value="0%")
        self.time_var = tk.StringVar(value="—")
        self.preview_var = tk.StringVar(value="Выберите файл и нажмите «Старт»")

        self._build_ui()
        self._select_preset("safe")
        self._select_engine("whisper")
        self.audio_var.trace_add("write", self._on_audio_path_change)
        self.bind_all("<Control-o>", lambda _event: self._pick_audio())
        self.bind_all("<Control-Return>", lambda _event: self._start())
        self.bind_all("<Escape>", lambda _event: self._cancel_run())
        self._poll_after_id = self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = ctk.CTkFrame(self, fg_color="transparent")
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
        self.left_scroll = left

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
            state="disabled",
        )
        self.start_btn.pack(side="left")

        self.cancel_btn = SecondaryButton(
            bar_inner,
            text="Отмена",
            width=120,
            command=self._cancel_run,
            state="disabled",
        )
        self.cancel_btn.pack(side="left", padx=(12, 0))

        self.open_btn = SecondaryButton(
            bar_inner,
            text="Открыть результаты",
            width=180,
            command=self._open_output,
            state="disabled",
        )
        self.open_btn.pack(side="left", padx=(12, 0))

        self.open_file_btn = SecondaryButton(
            bar_inner,
            text="Открыть текст",
            width=150,
            command=self._open_output_file,
            state="disabled",
        )
        self.open_file_btn.pack(side="left", padx=(12, 0))

        ctk.CTkLabel(
            bar_inner,
            textvariable=self.status_var,
            font=FONTS["label"],
            text_color=COLORS["text_muted"],
        ).pack(side="right")

        # Панель действий резервирует место первой и остаётся видимой при ресайзе.
        root.pack(fill="both", expand=True, padx=24, pady=20)

    def _build_left(self, parent) -> None:
        ctk.CTkLabel(
            parent,
            text="audio2text studio",
            font=FONTS["hero"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            parent,
            text="Локальная расшифровка без загрузки аудио в облако",
            font=FONTS["body"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 24))

        brand_row = ctk.CTkFrame(parent, fg_color="transparent")
        brand_row.pack(fill="x", pady=(0, 18))
        ctk.CTkLabel(
            brand_row,
            text="  100% ЛОКАЛЬНО  ",
            height=28,
            corner_radius=14,
            fg_color=COLORS["success_soft"],
            text_color=COLORS["success"],
            font=FONTS["small"],
        ).pack(side="left")
        ctk.CTkLabel(
            brand_row,
            text="by NikitaKHS",
            font=FONTS["caption"],
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(10, 0))

        hardware = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=RADIUS["md"],
            border_width=1,
            border_color=COLORS["border"],
        )
        hardware.pack(fill="x", pady=(0, 24))
        ctk.CTkLabel(
            hardware,
            textvariable=self.hardware_var,
            font=FONTS["label"],
            text_color=COLORS["text_secondary"],
            anchor="w",
            justify="left",
            wraplength=680,
        ).pack(fill="x", padx=16, pady=12)

        # ── Шаг 1 ──
        StepHeader(
            parent,
            1,
            "Выберите файл",
            "Укажите запись, которую нужно расшифровать",
        ).pack(fill="x", pady=(0, 10))

        sec1 = Section(parent)
        sec1.pack(fill="x", pady=(0, 28))
        BigFilePicker(
            sec1.body,
            self.audio_var,
            self._pick_audio,
        ).pack(fill="x")
        OutputFolderRow(
            sec1.body,
            self.output_var,
            self._pick_output,
        ).pack(fill="x", pady=(8, 0))

        # ── Шаг 2 ──
        StepHeader(
            parent,
            2,
            "Движок распознавания",
            "Whisper — быстрее; GigaAM — точнее для русского",
        ).pack(fill="x", pady=(0, 10))

        engine_section = Section(parent)
        engine_section.pack(fill="x", pady=(0, 28))
        engine_grid = ctk.CTkFrame(engine_section.body, fg_color="transparent")
        engine_grid.pack(fill="x")
        engine_grid.grid_columnconfigure((0, 1), weight=1, uniform="engine")
        for column, engine_id in enumerate(("whisper", "gigaam")):
            ui = ENGINE_UI[engine_id]
            card = PresetCard(
                engine_grid,
                preset_id=engine_id,
                title=ui["title"],
                description=ui["description"],
                meta=ui["meta"],
                badge=ui["badge"],
                on_select=self._select_engine,
            )
            card.grid(row=0, column=column, sticky="nsew", padx=6, pady=4)
            self._engine_cards[engine_id] = card

        # ── Шаг 3 ──
        StepHeader(
            parent,
            3,
            "Настройки режима",
            "Показываются только параметры выбранного движка",
        ).pack(fill="x", pady=(0, 10))

        self.mode_options = ctk.CTkFrame(parent, fg_color="transparent")
        self.mode_options.pack(fill="x")
        self.whisper_panel = Section(self.mode_options)
        self.whisper_panel.pack(fill="x", pady=(0, 28))

        ctk.CTkLabel(
            self.whisper_panel.body,
            text="Профиль Whisper",
            font=FONTS["body_bold"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 10))
        grid = ctk.CTkFrame(self.whisper_panel.body, fg_color="transparent")
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

        self.giga_panel = Section(self.mode_options)
        giga_head = ctk.CTkFrame(self.giga_panel.body, fg_color="transparent")
        giga_head.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(
            giga_head,
            text="Двойная проверка русского текста",
            font=FONTS["body_bold"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            giga_head,
            text=(
                "RNNT создаёт основной текст, CTC независимо проверяет каждый речевой фрагмент. "
                "Расхождения сохраняются в отдельный review-файл."
            ),
            font=FONTS["caption"],
            text_color=COLORS["text_secondary"],
            anchor="w",
            justify="left",
            wraplength=560,
        ).pack(anchor="w", pady=(5, 0))

        pipeline = ctk.CTkFrame(
            self.giga_panel.body,
            fg_color=COLORS["input"],
            corner_radius=RADIUS["md"],
            border_width=1,
            border_color=COLORS["border"],
        )
        pipeline.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(
            pipeline,
            text="РЕЧЬ  →  GigaAM RNNT  →  GigaAM CTC  →  СВЕРКА  →  TXT + SRT + REVIEW",
            font=FONTS["small"],
            text_color=COLORS["accent_hover"],
            anchor="w",
        ).pack(fill="x", padx=16, pady=13)
        ctk.CTkLabel(
            self.giga_panel.body,
            text="Все форматы и безопасное продолжение после сбоя включаются автоматически.",
            font=FONTS["caption"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 14))

        ctk.CTkLabel(
            self.giga_panel.body,
            text="Подтверждённые ручные исправления · необязательно",
            font=FONTS["label"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        correction_row = ctk.CTkFrame(self.giga_panel.body, fg_color="transparent")
        correction_row.pack(fill="x", pady=(7, 12))
        ctk.CTkEntry(
            correction_row,
            textvariable=self.corrections_var,
            height=44,
            placeholder_text="JSON с заменами — можно оставить пустым",
            font=FONTS["body"],
            fg_color=COLORS["elevated"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        SecondaryButton(
            correction_row,
            text="Выбрать JSON",
            width=140,
            height=44,
            command=self._pick_corrections,
        ).pack(side="right")

        self.giga_install_status_var = tk.StringVar()
        install_row = ctk.CTkFrame(self.giga_panel.body, fg_color="transparent")
        install_row.pack(fill="x")
        ctk.CTkLabel(
            install_row,
            textvariable=self.giga_install_status_var,
            font=FONTS["caption"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        self.install_accuracy_btn = SecondaryButton(
            install_row,
            text="Установить GigaAM-v3",
            width=190,
            height=44,
            command=self._install_accuracy,
        )
        self.install_accuracy_btn.pack(side="right")

        # ── Шаг 4 ──
        StepHeader(
            parent,
            4,
            "Результаты и надёжность",
            "Для длинного аудио оставьте все пункты включёнными",
        ).pack(fill="x", pady=(0, 10))

        sec3 = Section(parent)
        sec3.pack(fill="x", pady=(0, 28))

        toggles = [
            ("Очистка текста", "Убирает повторы и мусор Whisper", self.postprocess_var),
            ("Субтитры SRT", "Файл с таймкодами для видеоплеера", self.save_srt_var),
            ("Текст без меток", "Сплошной текст для чтения", self.save_plain_var),
            ("Черновик raw", "До очистки — для отладки", self.save_raw_var),
            (
                "Список для проверки",
                "Неуверенные фрагменты для ручной сверки",
                self.save_review_var,
            ),
            (
                "Продолжать после сбоя",
                "Сохраняет прогресс и продолжает с последнего таймкода",
                self.resume_var,
            ),
        ]
        self.result_toggle_rows: list[ToggleRow] = []
        for title, sub, var in toggles:
            row = ToggleRow(sec3.body, title, sub, var)
            row.pack(fill="x", pady=5)
            self.result_toggle_rows.append(row)

        # ── Расширенные ──
        adv = Collapsible(parent, "Расширенные настройки (модель, Whisper, контекст)")
        adv.pack(fill="x", pady=(0, 16))
        self.advanced_panel = adv

        row_m = ctk.CTkFrame(adv.body, fg_color="transparent")
        row_m.pack(fill="x", pady=(0, 12))
        row_m.grid_columnconfigure((0, 1, 2), weight=1)

        FieldCombo(
            row_m,
            "Модель",
            "Whisper-модель",
            self.model_var,
            MODELS,
            readonly=False,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldCombo(
            row_m,
            "Устройство",
            "GPU быстрее",
            self.device_var,
            DEVICES,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        FieldCombo(
            row_m,
            "Номер GPU",
            "Для нескольких видеокарт",
            self.device_index_var,
            [str(gpu.index) for gpu in self._gpus] or ["0"],
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        row_m2 = ctk.CTkFrame(adv.body, fg_color="transparent")
        row_m2.pack(fill="x", pady=(0, 12))
        row_m2.grid_columnconfigure((0, 1), weight=1)
        FieldCombo(
            row_m2,
            "Точность VRAM",
            "float16 — стандарт",
            self.compute_var,
            COMPUTE_TYPES,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        FieldCombo(
            row_m2,
            "Язык",
            "ru для русского",
            self.language_var,
            ["ru", "en", "uk", "de", "fr"],
            readonly=False,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row_p = ctk.CTkFrame(adv.body, fg_color="transparent")
        row_p.pack(fill="x", pady=(0, 12))
        row_p.grid_columnconfigure((0, 1, 2), weight=1)
        FieldEntry(
            row_p,
            "Глубина поиска",
            "beam_size · 5–10 для качества",
            self.beam_var,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldEntry(
            row_p,
            "Кандидатов",
            "best_of",
            self.best_of_var,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        FieldEntry(
            row_p,
            "Порог тишины",
            "0.7–0.8 для длинных записей",
            self.no_speech_var,
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
            adv.body,
            text="Контекст для модели",
            font=FONTS["label"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w", pady=(16, 4))
        ctk.CTkLabel(
            adv.body,
            text="Опишите тему разговора — имена, термины, обстановку",
            font=FONTS["caption"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        self.prompt_entry = ctk.CTkTextbox(
            adv.body,
            height=100,
            font=FONTS["body"],
            fg_color=COLORS["input"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=RADIUS["sm"],
            text_color=COLORS["text"],
        )
        self.prompt_entry.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            adv.body,
            text="Ключевые слова",
            font=FONTS["label"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 4))
        self.hotwords_entry = ctk.CTkTextbox(
            adv.body,
            height=64,
            font=FONTS["body"],
            fg_color=COLORS["input"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=RADIUS["sm"],
            text_color=COLORS["text"],
        )
        self.hotwords_entry.pack(fill="x")

    def _build_right(self, parent) -> None:
        pad = ctk.CTkFrame(parent, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=22, pady=22)

        title_row = ctk.CTkFrame(pad, fg_color="transparent")
        title_row.pack(fill="x")
        ctk.CTkLabel(
            title_row,
            text="Прогресс",
            font=FONTS["h1"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            title_row,
            textvariable=self.engine_badge_var,
            height=28,
            corner_radius=14,
            fg_color=COLORS["accent_soft"],
            text_color=COLORS["accent_hover"],
            font=FONTS["small"],
        ).pack(side="right")

        stat_row = ctk.CTkFrame(pad, fg_color="transparent")
        stat_row.pack(fill="x", pady=(20, 8))
        ctk.CTkLabel(
            stat_row,
            textvariable=self.progress_pct_var,
            font=FONTS["stat"],
            text_color=COLORS["accent_hover"],
        ).pack(side="left")
        ctk.CTkLabel(
            stat_row,
            textvariable=self.time_var,
            font=FONTS["h2"],
            text_color=COLORS["text_muted"],
        ).pack(side="right", pady=(8, 0))

        self.progress = ctk.CTkProgressBar(
            pad,
            height=10,
            corner_radius=5,
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
            pad,
            text="Журнал",
            font=FONTS["h2"],
            text_color=COLORS["text"],
            anchor="w",
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

    def _select_engine(self, engine_id: str) -> None:
        if self._worker and self._worker.is_alive():
            return
        if engine_id not in ENGINE_UI:
            return
        self.engine_var.set(engine_id)
        for key, card in self._engine_cards.items():
            card.set_selected(key == engine_id)

        if engine_id == "gigaam":
            self.whisper_panel.pack_forget()
            self.giga_panel.pack(fill="x", pady=(0, 28))
            self.advanced_panel.pack_forget()
            self.engine_badge_var.set("GIGAAM · RNNT + CTC")
            self.start_btn.configure(text="  Запустить двойную проверку  ")
            installed = accuracy_installed()
            self.giga_install_status_var.set(
                "Компоненты установлены · модели работают локально"
                if installed
                else "Требуется однократная установка компонентов (~1,5 ГБ)"
            )
            self.install_accuracy_btn.configure(
                state="disabled" if installed else "normal",
                text="GigaAM-v3 установлен" if installed else "Установить GigaAM-v3",
            )
            self.preview_var.set("GigaAM проверит русскую речь двумя декодерами")
            for variable in (
                self.postprocess_var,
                self.save_srt_var,
                self.save_plain_var,
                self.save_raw_var,
                self.save_review_var,
                self.resume_var,
            ):
                variable.set(True)
            for row in self.result_toggle_rows:
                row.set_enabled(False)
        else:
            self.giga_panel.pack_forget()
            self.whisper_panel.pack(fill="x", pady=(0, 28))
            if not self.advanced_panel.winfo_manager():
                self.advanced_panel.pack(fill="x", pady=(0, 16))
            self.engine_badge_var.set("WHISPER GPU")
            self.start_btn.configure(text="  Начать транскрибацию  ")
            self.preview_var.set("Whisper large-v3 готов к работе")
            for row in self.result_toggle_rows:
                row.set_enabled(True)

    def _pick_corrections(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        path = filedialog.askopenfilename(
            title="JSON с подтверждёнными исправлениями",
            filetypes=[("JSON", "*.json"), ("Все файлы", "*.*")],
        )
        if path:
            self.corrections_var.set(path)

    def _install_accuracy(self) -> None:
        script = Path(__file__).with_name("setup_accuracy.bat")
        if not script.is_file():
            messagebox.showerror("audio2text", f"Установщик не найден:\n{script}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(script)  # type: ignore[attr-defined]
            else:
                messagebox.showinfo(
                    "Установка GigaAM-v3",
                    "Выполните: python -m pip install -r requirements-accuracy.txt",
                )
                return
        except OSError as exc:
            messagebox.showerror("audio2text", f"Не удалось запустить установщик:\n{exc}")
            return
        messagebox.showinfo(
            "Установка запущена",
            "Дождитесь сообщения «Готово» в окне установщика, затем перезапустите audio2text.",
        )

    def _on_audio_path_change(self, *_args) -> None:
        if self._worker and self._worker.is_alive():
            return
        path_text = self.audio_var.get().strip()
        valid = bool(path_text) and Path(path_text).is_file()
        self.start_btn.configure(state="normal" if valid else "disabled")
        if valid:
            path = Path(path_text)
            self.preview_var.set(f"Готово к запуску: {path.name}")
            self.status_var.set("Файл выбран — можно начинать")

    def _select_preset(self, name: str) -> None:
        if self._worker and self._worker.is_alive():
            return
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
        if self._worker and self._worker.is_alive():
            return
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
        if self._worker and self._worker.is_alive():
            return
        path = filedialog.askdirectory(title="Папка для результатов")
        if path:
            self.output_var.set(path)

    def _append_log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _begin_run(self, label: str) -> None:
        self._cancel = False
        self._clear_log()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.open_file_btn.configure(state="disabled")
        self.progress.set(0)
        self.progress_pct_var.set("0%")
        self.time_var.set("—")
        self.status_var.set("Подготовка…")
        self.preview_var.set(label)

    def _build_config(self) -> TranscribeConfig:
        audio = Path(self.audio_var.get().strip())
        out = self.output_var.get().strip()
        cfg = TranscribeConfig(
            audio_path=audio,
            output_dir=Path(out) if out else None,
            model=self.model_var.get(),
            device=self.device_var.get(),
            device_index=int(self.device_index_var.get()),
            compute_type=self.compute_var.get(),
            language=self.language_var.get(),
            preset=self.preset_var.get(),
            initial_prompt=self.prompt_entry.get("1.0", "end").strip(),
            hotwords=self.hotwords_entry.get("1.0", "end").strip(),
            postprocess=self.postprocess_var.get(),
            save_srt=self.save_srt_var.get(),
            save_plain=self.save_plain_var.get(),
            save_raw=self.save_raw_var.get(),
            save_review=self.save_review_var.get(),
            resume=self.resume_var.get(),
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
                "patience",
                "temperature",
                "vad_threshold",
                "vad_min_speech_ms",
                "vad_min_silence_ms",
                "word_timestamps",
                "compression_ratio_threshold",
                "hallucination_silence_threshold",
                "low_confidence_logprob",
            ):
                setattr(cfg.params, field, getattr(base, field))
        cfg.validate()
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

        if self.engine_var.get() == "gigaam":
            self._start_accuracy(Path(audio))
            return

        try:
            cfg = self._build_config()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Некорректные настройки", str(exc))
            return

        selected_gpu = next(
            (gpu for gpu in self._gpus if gpu.index == cfg.device_index and gpu.runtime_ready),
            None,
        )
        if cfg.device != "cpu":
            warning = vram_warning(cfg.model, cfg.effective_compute_type(), selected_gpu)
            if warning and not messagebox.askyesno(
                "Риск нехватки видеопамяти",
                f"{warning}\n\nВсё равно запустить?",
                icon="warning",
            ):
                return

        self._begin_run("Whisper загружает модель и анализирует файл")
        self._append_log(f"[Старт] {cfg.audio_path.name} | {PRESET_UI[cfg.preset]['title']}")

        def run() -> None:
            try:
                paths = transcribe_file(
                    cfg,
                    log=lambda m: self._msg_queue.put(("log", m)),
                    progress=lambda s, sn: self._msg_queue.put(("progress", (s, sn))),
                    cancel_check=lambda: self._cancel,
                )
                self._msg_queue.put(
                    (
                        "done",
                        {
                            "output_dir": str(paths["final_txt"].parent.resolve()),
                            "output_file": str(paths["final_txt"].resolve()),
                            "message": "Whisper-транскрибация завершена",
                        },
                    )
                )
            except InterruptedError:
                self._msg_queue.put(("cancelled", None))
            except Exception as e:
                self._msg_queue.put(("error", str(e)))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _start_accuracy(self, audio: Path) -> None:
        if not accuracy_installed():
            messagebox.showwarning(
                "Нужна установка GigaAM-v3",
                "Нажмите «Установить GigaAM-v3», дождитесь завершения и перезапустите приложение.",
            )
            return
        output_text = self.output_var.get().strip()
        output_dir = Path(output_text) if output_text else None
        corrections_text = self.corrections_var.get().strip()
        corrections = Path(corrections_text) if corrections_text else None
        if corrections is not None and not corrections.is_file():
            messagebox.showerror("audio2text", f"JSON исправлений не найден:\n{corrections}")
            return

        self._begin_run("GigaAM готовит речевые фрагменты")
        self._append_log(f"[Старт] {audio.name} | GigaAM-v3 RNNT + CTC")
        self._append_log("Режим: русский · двойная локальная проверка")

        def run() -> None:
            try:
                paths = transcribe_high_accuracy(
                    audio,
                    output_dir=output_dir,
                    corrections_path=corrections,
                    progress=lambda done, total, sec: self._msg_queue.put(
                        ("accuracy_progress", (done, total, sec))
                    ),
                    log=lambda message: self._msg_queue.put(("log", message)),
                    cancel_check=lambda: self._cancel,
                )
                self._msg_queue.put(
                    (
                        "done",
                        {
                            "output_dir": str(paths["txt"].parent.resolve()),
                            "output_file": str(paths["txt"].resolve()),
                            "message": "Двойная проверка GigaAM завершена",
                        },
                    )
                )
            except InterruptedError:
                self._msg_queue.put(("cancelled", None))
            except Exception as exc:
                self._msg_queue.put(("error", str(exc)))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _cancel_run(self) -> None:
        if not self._worker or not self._worker.is_alive():
            return
        self._cancel = True
        self.status_var.set("Отмена…")
        self.preview_var.set("Завершаем текущий фрагмент и сохраняем checkpoint")

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
                elif kind == "accuracy_progress":
                    done, total, sec = payload  # type: ignore[misc]
                    pct = min(1.0, done / total) if total else 0
                    self.progress.set(pct)
                    self.progress_pct_var.set(f"{int(pct * 100)}%")
                    self.time_var.set(f"{done}/{total} · до {_fmt_time(sec)}")
                    self.preview_var.set(f"RNNT + CTC сверяют фрагмент {done} из {total}")
                    self.status_var.set("Двойная проверка текста…")
                elif kind == "done":
                    result = payload  # type: ignore[assignment]
                    self._finish(
                        success=True,
                        msg=str(result.get("message", "Готово")),
                        output_dir=str(result.get("output_dir", "")),
                        output_file=str(result.get("output_file", "")),
                    )
                elif kind == "cancelled":
                    self._finish(success=False, msg="Отменено — прогресс сохранён")
                elif kind == "error":
                    self._finish(success=False, msg="Ошибка")
                    messagebox.showerror("Ошибка", str(payload))
        except queue.Empty:
            pass
        self._poll_after_id = self.after(100, self._poll_queue)

    def _finish(
        self,
        success: bool,
        msg: str = "Готово",
        output_dir: str | None = None,
        output_file: str | None = None,
    ) -> None:
        audio_exists = Path(self.audio_var.get().strip()).is_file()
        self.start_btn.configure(state="normal" if audio_exists else "disabled")
        self.cancel_btn.configure(state="disabled")
        self.status_var.set(msg)
        if success:
            self.progress.set(1.0)
            self.progress_pct_var.set("100%")
            self.preview_var.set("Транскрибация завершена")
            self._append_log("[Готово] Все файлы сохранены")
            if output_dir:
                self._last_output_dir = Path(output_dir)
                self.open_btn.configure(state="normal")
            if output_file:
                self._last_output_file = Path(output_file)
                self.open_file_btn.configure(state="normal")

    def _open_output(self) -> None:
        path = getattr(self, "_last_output_dir", None)
        if not path or not Path(path).is_dir():
            messagebox.showerror("audio2text", "Папка результатов не найдена")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("audio2text", f"Не удалось открыть папку:\n{exc}")

    def _open_output_file(self) -> None:
        path = getattr(self, "_last_output_file", None)
        if not path or not Path(path).is_file():
            messagebox.showerror("audio2text", "Итоговый текст не найден")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("audio2text", f"Не удалось открыть текст:\n{exc}")

    def destroy(self) -> None:
        """Останавливает worker и удаляет все Tk-callback'и перед закрытием окна."""
        self._cancel = True
        try:
            self.after_cancel(self._poll_after_id)
        except (AttributeError, tk.TclError):
            pass
        super().destroy()


# Re-export for tests
PRESET_LABELS = {k: v["title"] for k, v in PRESET_UI.items()}
LABEL_TO_PRESET = {v["title"]: k for k, v in PRESET_UI.items()}


def main() -> None:
    app = TranscribeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
