"""Тесты постобработки."""
import unittest

from a2t_lib.postprocess import (
    Segment,
    clean_segments,
    is_junk,
    is_live_spam,
    is_single_word_spam,
    keep_legitimate_context,
    parse_transcript_file,
    remove_loops,
    segments_to_srt,
    segments_to_txt,
)


class TestPostprocess(unittest.TestCase):
    def test_remove_loops(self):
        text = "ну, и потом, вот, и потом, вот, и потом, вот"
        cleaned = remove_loops(text)
        self.assertNotIn("и потом, вот, и потом", cleaned)

    def test_is_junk_youtube_phrase(self):
        seg = Segment(0, 1, "Спасибо за просмотр, подписывайтесь!")
        junk, reason = is_junk(seg)
        self.assertTrue(junk)
        self.assertIn("галлюцинация", reason)

    def test_is_live_spam_repeated_word(self):
        self.assertFalse(is_live_spam("Спасибо.", 1))
        self.assertTrue(is_live_spam("Спасибо.", 3))

    def test_is_live_spam_generic_repeat(self):
        self.assertTrue(is_live_spam("да да", 4))

    def test_keep_legitimate_context_after_question(self):
        segments = [
            Segment(10, 12, "А чего ты боялся?"),
            Segment(12, 13, "Умереть."),
        ]
        self.assertTrue(keep_legitimate_context(segments, 1))

    def test_spam_without_context_removed(self):
        segments = [
            Segment(0, 1, "Тишина."),
            Segment(100, 101, "Умереть."),
            Segment(101, 102, "Умереть."),
        ]
        cleaned, log = clean_segments(segments)
        self.assertEqual(len(cleaned), 1)
        self.assertIn("однословный", "".join(log))

    def test_legitimate_answer_kept_in_clean(self):
        segments = [
            Segment(10, 12, "Чего боялся?"),
            Segment(12, 13, "Умереть."),
        ]
        cleaned, _ = clean_segments(segments)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[1].text, "Умереть.")

    def test_segments_to_srt_format(self):
        segs = [Segment(0, 1.5, "Тест")]
        srt = segments_to_srt(segs)
        self.assertIn("1\n", srt)
        self.assertIn("00:00:00,000 --> 00:00:01,500", srt)
        self.assertIn("Тест", srt)

    def test_segments_to_txt_format(self):
        segs = [Segment(0, 1, "Привет")]
        txt = segments_to_txt(segs)
        self.assertIn("[00:00:00.000 -> 00:00:01.000] Привет", txt)

    def test_parse_transcript_file(self):
        from pathlib import Path
        import tempfile

        content = (
            "[00:00:01.000 -> 00:00:02.000] Первая реплика\n\n"
            "[00:00:03.000 -> 00:00:04.000] Вторая реплика"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        ) as f:
            f.write(content)
            path = Path(f.name)
        try:
            segs = parse_transcript_file(path)
            self.assertEqual(len(segs), 2)
            self.assertEqual(segs[0].text, "Первая реплика")
        finally:
            path.unlink(missing_ok=True)

    def test_is_single_word_spam(self):
        self.assertTrue(is_single_word_spam("Спасибо."))
        self.assertFalse(is_single_word_spam("Нормальный длинный текст"))


if __name__ == "__main__":
    unittest.main()
