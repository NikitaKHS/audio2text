"""Smoke-тест GUI (создание окна без mainloop)."""

import unittest


class TestGUISmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transcribe_gui import TranscribeApp

        cls.app = TranscribeApp()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def test_app_creates_and_builds_config(self):
        self.app.audio_var.set("test.m4a")
        self.app.preset_var.set("safe")
        self.app.prompt_entry.delete("1.0", "end")
        self.app.prompt_entry.insert("1.0", "Тестовый prompt")
        self.app.hotwords_entry.delete("1.0", "end")
        self.app.hotwords_entry.insert("1.0", "слово1 слово2")
        cfg = self.app._build_config()
        self.assertEqual(cfg.preset, "safe")
        self.assertEqual(cfg.initial_prompt, "Тестовый prompt")
        self.assertEqual(cfg.hotwords, "слово1 слово2")
        self.assertFalse(cfg.params.condition_on_previous_text)
        self.assertEqual(cfg.device_index, 0)
        self.assertTrue(cfg.resume)

    def test_preset_change_updates_fields(self):
        self.app._select_preset("max_quality")
        self.assertEqual(self.app.model_var.get(), "large-v3")
        self.assertEqual(self.app.beam_var.get(), "10")

    def test_preset_segment_mapping(self):
        from transcribe_gui import LABEL_TO_PRESET, PRESET_LABELS

        self.assertEqual(LABEL_TO_PRESET[PRESET_LABELS["fast"]], "fast")

    def test_engine_switch_updates_visible_mode(self):
        self.app._select_engine("gigaam")
        self.assertEqual(self.app.engine_var.get(), "gigaam")
        self.assertIn("GIGAAM", self.app.engine_badge_var.get())
        self.assertIn("двойную проверку", self.app.start_btn.cget("text").lower())
        self.assertEqual(self.app.whisper_panel.winfo_manager(), "")

        self.app._select_engine("whisper")
        self.assertEqual(self.app.engine_var.get(), "whisper")
        self.assertIn("WHISPER", self.app.engine_badge_var.get())

    def test_engine_card_description_is_clickable(self):
        self.app._select_engine("whisper")
        giga_card = self.app._engine_cards["gigaam"]

        # CTkLabel renders its text through the internal Tk label; this is the
        # exact surface that receives a real mouse click.
        giga_card.description_lbl._label.event_generate("<Button-1>")
        self.app.update()

        self.assertEqual(self.app.engine_var.get(), "gigaam")


if __name__ == "__main__":
    unittest.main()
