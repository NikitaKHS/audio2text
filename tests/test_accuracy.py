import unittest

from a2t_lib.accuracy import ManualCorrection, apply_manual_corrections, text_agreement
from a2t_lib.postprocess import Segment


class TestAccuracy(unittest.TestCase):
    def test_text_agreement_ignores_case_punctuation_and_yo(self):
        self.assertEqual(text_agreement("Всё, хорошо!", "все хорошо"), 1.0)

    def test_text_agreement_detects_different_hypotheses(self):
        self.assertLess(text_agreement("происходит в стране", "мы во сне"), 0.5)

    def test_manual_correction_is_limited_by_time_and_text(self):
        segments = [
            Segment(0, 4, "другой текст"),
            Segment(5, 10, "ошибочная фраза"),
        ]
        corrections = [ManualCorrection(4, 11, "ошибочная", "верная", "проверено")]

        corrected, log = apply_manual_corrections(segments, corrections)

        self.assertEqual(corrected[0].text, "другой текст")
        self.assertEqual(corrected[1].text, "верная фраза")
        self.assertIn("проверено", log[0])


if __name__ == "__main__":
    unittest.main()
