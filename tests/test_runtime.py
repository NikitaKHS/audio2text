"""Тесты системных гарантий длительной задачи."""

import unittest
from unittest.mock import patch

from a2t_lib.runtime import prevent_system_sleep


class TestRuntime(unittest.TestCase):
    def test_sleep_prevention_is_always_restored(self):
        @prevent_system_sleep
        def fail() -> None:
            raise RuntimeError("test")

        with patch("a2t_lib.runtime._set_sleep_prevention") as set_state:
            with self.assertRaises(RuntimeError):
                fail()
        self.assertEqual([call.args[0] for call in set_state.call_args_list], [True, False])


if __name__ == "__main__":
    unittest.main()
