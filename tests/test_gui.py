"""Smoke-тест GUI (создание окна без mainloop)."""
import unittest


class TestGUISmoke(unittest.TestCase):
    def test_app_creates_and_builds_config(self):
        from transcribe_gui import TranscribeApp

        app = TranscribeApp()
        try:
            app.audio_var.set("test.m4a")
            app.preset_var.set("safe")
            app.prompt_entry.delete("1.0", "end")
            app.prompt_entry.insert("1.0", "Тестовый prompt")
            app.hotwords_entry.delete("1.0", "end")
            app.hotwords_entry.insert("1.0", "слово1 слово2")
            cfg = app._build_config()
            self.assertEqual(cfg.preset, "safe")
            self.assertEqual(cfg.initial_prompt, "Тестовый prompt")
            self.assertEqual(cfg.hotwords, "слово1 слово2")
            self.assertFalse(cfg.params.condition_on_previous_text)
        finally:
            app.destroy()

    def test_preset_change_updates_fields(self):
        from transcribe_gui import TranscribeApp

        app = TranscribeApp()
        try:
            app._select_preset("max_quality")
            self.assertEqual(app.model_var.get(), "large-v3")
            self.assertEqual(app.beam_var.get(), "10")
        finally:
            app.destroy()

    def test_preset_segment_mapping(self):
        from transcribe_gui import LABEL_TO_PRESET, PRESET_LABELS

        self.assertEqual(LABEL_TO_PRESET[PRESET_LABELS["fast"]], "fast")


if __name__ == "__main__":
    unittest.main()
