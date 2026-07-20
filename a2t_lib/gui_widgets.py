"""Виджеты CustomTkinter. Цвета и размеры — константы в начале файла."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

# ── Design system ────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#08080a",
    "panel": "#0f0f13",
    "surface": "#16161c",
    "surface_hover": "#1f1f28",
    "elevated": "#26262f",
    "border": "#33333f",
    "border_active": "#6366f1",
    "text": "#fafafa",
    "text_secondary": "#c4c4cc",
    "text_muted": "#8b8b98",
    "accent": "#6366f1",
    "accent_hover": "#818cf8",
    "accent_glow": "#4f46e5",
    "accent_soft": "#1e1b4b",
    "success": "#22c55e",
    "success_soft": "#14532d",
    "danger": "#ef4444",
    "danger_soft": "#450a0a",
    "input": "#0c0c10",
    "log": "#060608",
}

RADIUS = {"sm": 10, "md": 14, "lg": 18, "xl": 22}

FONTS = {
    "hero": ("Segoe UI", 32, "bold"),
    "h1": ("Segoe UI", 20, "bold"),
    "h2": ("Segoe UI", 16, "bold"),
    "body": ("Segoe UI", 15),
    "body_bold": ("Segoe UI", 15, "bold"),
    "label": ("Segoe UI", 14),
    "caption": ("Segoe UI", 13),
    "small": ("Segoe UI", 12),
    "mono": ("Consolas", 14),
    "button_lg": ("Segoe UI", 16, "bold"),
    "button": ("Segoe UI", 14, "bold"),
    "stat": ("Segoe UI", 28, "bold"),
}

# CustomTkinter scaling — на 1080p без этого мелковато
SCALE = 1.18


def apply_theme() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    ctk.set_widget_scaling(SCALE)
    ctk.set_window_scaling(1.0)


class StepHeader(ctk.CTkFrame):
    """Заголовок шага: номер + название + описание."""

    def __init__(
        self,
        master,
        step: int,
        title: str,
        subtitle: str = "",
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")

        badge = ctk.CTkLabel(
            row,
            text=str(step),
            width=36,
            height=36,
            corner_radius=18,
            fg_color=COLORS["accent"],
            text_color="#ffffff",
            font=FONTS["body_bold"],
        )
        badge.pack(side="left")

        texts = ctk.CTkFrame(row, fg_color="transparent")
        texts.pack(side="left", fill="x", expand=True, padx=(14, 0))
        ctk.CTkLabel(
            texts,
            text=title,
            font=FONTS["h2"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        if subtitle:
            ctk.CTkLabel(
                texts,
                text=subtitle,
                font=FONTS["caption"],
                text_color=COLORS["text_muted"],
                anchor="w",
            ).pack(anchor="w", pady=(2, 0))


class Section(ctk.CTkFrame):
    """Блок контента под шагом."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(
            master,
            fg_color=COLORS["surface"],
            corner_radius=RADIUS["lg"],
            border_width=1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=22, pady=20)


class BigFilePicker(ctk.CTkFrame):
    """Крупная зона выбора файла."""

    def __init__(
        self,
        master,
        variable: tk.StringVar,
        pick_cmd: Callable[[], None],
        pick_folder_cmd: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)

        # Аудио
        audio_box = ctk.CTkFrame(
            self,
            fg_color=COLORS["input"],
            corner_radius=RADIUS["md"],
            border_width=2,
            border_color=COLORS["border"],
        )
        audio_box.pack(fill="x", pady=(0, 14))

        inner = ctk.CTkFrame(audio_box, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            inner,
            text="Аудиофайл",
            font=FONTS["body_bold"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner,
            text="M4A, MP3, WAV, OGG, FLAC — до нескольких часов",
            font=FONTS["caption"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(4, 12))

        row = ctk.CTkFrame(inner, fg_color="transparent")
        row.pack(fill="x")

        self.audio_entry = ctk.CTkEntry(
            row,
            textvariable=variable,
            height=48,
            placeholder_text="Нажмите «Выбрать файл» или вставьте путь…",
            font=FONTS["body"],
            corner_radius=RADIUS["sm"],
            fg_color=COLORS["elevated"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            placeholder_text_color=COLORS["text_muted"],
        )
        self.audio_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkButton(
            row,
            text="Выбрать файл",
            width=160,
            height=48,
            font=FONTS["button"],
            corner_radius=RADIUS["sm"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=pick_cmd,
        ).pack(side="right")


class OutputFolderRow(ctk.CTkFrame):
    def __init__(
        self,
        master,
        variable: tk.StringVar,
        pick_cmd: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        ctk.CTkLabel(
            self,
            text="Куда сохранить результат",
            font=FONTS["label"],
            text_color=COLORS["text_secondary"],
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkEntry(
            row,
            textvariable=variable,
            height=44,
            placeholder_text="Пусто — в ту же папку, где лежит аудио",
            font=FONTS["body"],
            corner_radius=RADIUS["sm"],
            fg_color=COLORS["elevated"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            placeholder_text_color=COLORS["text_muted"],
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(
            row,
            text="Папка",
            width=110,
            height=44,
            font=FONTS["button"],
            corner_radius=RADIUS["sm"],
            fg_color=COLORS["elevated"],
            hover_color=COLORS["surface_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            command=pick_cmd,
        ).pack(side="right")


class PresetCard(ctk.CTkFrame):
    """Кликабельная карточка пресета."""

    def __init__(
        self,
        master,
        preset_id: str,
        title: str,
        description: str,
        meta: str,
        on_select: Callable[[str], None],
        badge: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            master,
            fg_color=COLORS["input"],
            corner_radius=RADIUS["md"],
            border_width=2,
            border_color=COLORS["border"],
            cursor="hand2",
            **kwargs,
        )
        self.preset_id = preset_id
        self._on_select = on_select
        self._selected = False

        pad = ctk.CTkFrame(self, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=16, pady=16)

        title_row = ctk.CTkFrame(pad, fg_color="transparent")
        title_row.pack(fill="x")

        self.title_lbl = ctk.CTkLabel(
            title_row,
            text=title,
            font=FONTS["body_bold"],
            text_color=COLORS["text"],
            anchor="w",
        )
        self.title_lbl.pack(side="left", anchor="w")
        if badge:
            badge_label = ctk.CTkLabel(
                title_row,
                text=f"  {badge}  ",
                height=24,
                corner_radius=12,
                fg_color=COLORS["accent_soft"],
                text_color=COLORS["accent_hover"],
                font=FONTS["small"],
            )
            badge_label.pack(side="right")

        self.description_lbl = ctk.CTkLabel(
            pad,
            text=description,
            font=FONTS["caption"],
            text_color=COLORS["text_secondary"],
            anchor="w",
            wraplength=220,
            justify="left",
        )
        self.description_lbl.pack(anchor="w", pady=(6, 0))

        self.meta_lbl = ctk.CTkLabel(
            pad,
            text=meta,
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w",
            justify="left",
            wraplength=240,
        )
        self.meta_lbl.pack(anchor="w", pady=(10, 0))

        # CustomTkinter composes a card from nested frames, canvases and labels.
        # Binding only the outer frame leaves most of the visible card inert.
        self._bind_click_tree(self)

    def _bind_click_tree(self, widget: tk.Misc) -> None:
        widget.bind("<Button-1>", self._click, add="+")
        try:
            widget.configure(cursor="hand2")
        except (tk.TclError, TypeError, ValueError):
            pass
        for child in widget.winfo_children():
            self._bind_click_tree(child)

    def _click(self, _event=None) -> None:
        self._on_select(self.preset_id)

    def set_selected(self, yes: bool) -> None:
        self._selected = yes
        color = COLORS["border_active"] if yes else COLORS["border"]
        bg = COLORS["accent_soft"] if yes else COLORS["input"]
        self.configure(border_color=color, fg_color=bg)


class ToggleRow(ctk.CTkFrame):
    """Крупный переключатель с понятной подписью."""

    def __init__(
        self,
        master,
        title: str,
        subtitle: str,
        variable: tk.BooleanVar,
        **kwargs,
    ) -> None:
        super().__init__(
            master,
            fg_color=COLORS["input"],
            corner_radius=RADIUS["sm"],
            **kwargs,
        )
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=14)

        texts = ctk.CTkFrame(inner, fg_color="transparent")
        texts.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            texts,
            text=title,
            font=FONTS["label"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            texts,
            text=subtitle,
            font=FONTS["caption"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        self.switch = ctk.CTkSwitch(
            inner,
            text="",
            variable=variable,
            width=52,
            height=28,
            switch_width=48,
            switch_height=26,
            corner_radius=13,
            fg_color=COLORS["elevated"],
            progress_color=COLORS["accent"],
            button_color="#ffffff",
            button_hover_color="#ffffff",
        )
        self.switch.pack(side="right", padx=(12, 0))

    def set_enabled(self, enabled: bool) -> None:
        self.switch.configure(state="normal" if enabled else "disabled")


class FieldCombo(ctk.CTkFrame):
    def __init__(
        self,
        master,
        label: str,
        hint: str,
        variable: tk.StringVar,
        values: list[str],
        readonly: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        ctk.CTkLabel(
            self,
            text=label,
            font=FONTS["label"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        if hint:
            ctk.CTkLabel(
                self,
                text=hint,
                font=FONTS["small"],
                text_color=COLORS["text_muted"],
                anchor="w",
            ).pack(anchor="w", pady=(2, 8))
        else:
            ctk.CTkFrame(self, height=8, fg_color="transparent").pack()
        ctk.CTkComboBox(
            self,
            variable=variable,
            values=values,
            height=44,
            font=FONTS["body"],
            corner_radius=RADIUS["sm"],
            fg_color=COLORS["elevated"],
            border_color=COLORS["border"],
            button_color=COLORS["surface_hover"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["elevated"],
            text_color=COLORS["text"],
            state="readonly" if readonly else "normal",
        ).pack(fill="x")


class FieldEntry(ctk.CTkFrame):
    def __init__(
        self,
        master,
        label: str,
        hint: str,
        variable: tk.Variable,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        ctk.CTkLabel(
            self,
            text=label,
            font=FONTS["label"],
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            self,
            text=hint,
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w",
        ).pack(anchor="w", pady=(2, 8))
        ctk.CTkEntry(
            self,
            textvariable=variable,
            height=44,
            font=FONTS["body"],
            corner_radius=RADIUS["sm"],
            fg_color=COLORS["elevated"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
        ).pack(fill="x")


class Collapsible(ctk.CTkFrame):
    def __init__(self, master, title: str, **kwargs) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self._open = False
        self.btn = ctk.CTkButton(
            self,
            text=f"  >  {title}",
            anchor="w",
            height=44,
            font=FONTS["label"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["surface_hover"],
            text_color=COLORS["text_secondary"],
            corner_radius=RADIUS["sm"],
            command=self.toggle,
        )
        self.btn.pack(fill="x")
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content_body = ctk.CTkFrame(
            self.content,
            fg_color=COLORS["surface"],
            corner_radius=RADIUS["md"],
            border_width=1,
            border_color=COLORS["border"],
        )
        self.content_body.pack(fill="x", pady=(8, 0), padx=4)
        self.inner = ctk.CTkFrame(self.content_body, fg_color="transparent")
        self.inner.pack(fill="x", padx=18, pady=18)

    def toggle(self) -> None:
        self._open = not self._open
        title = self.btn.cget("text").strip(" >v").strip()
        if self._open:
            self.btn.configure(text=f"  v  {title.split('  ', 1)[-1]}")
            self.content.pack(fill="x", pady=(0, 8))
        else:
            self.btn.configure(text=f"  >  {title.split('  ', 1)[-1]}")
            self.content.pack_forget()

    @property
    def body(self) -> ctk.CTkFrame:
        return self.inner


class PrimaryButton(ctk.CTkButton):
    def __init__(self, master, **kwargs) -> None:
        defaults = {
            "height": 56,
            "corner_radius": RADIUS["md"],
            "fg_color": COLORS["accent"],
            "hover_color": COLORS["accent_hover"],
            "text_color": "#ffffff",
            "font": FONTS["button_lg"],
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)


class SecondaryButton(ctk.CTkButton):
    def __init__(self, master, **kwargs) -> None:
        defaults = {
            "height": 48,
            "corner_radius": RADIUS["md"],
            "fg_color": COLORS["elevated"],
            "hover_color": COLORS["surface_hover"],
            "border_width": 1,
            "border_color": COLORS["border"],
            "text_color": COLORS["text"],
            "font": FONTS["button"],
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)
