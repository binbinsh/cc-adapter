import unittest

from cc_adapter.config import Settings
from cc_adapter.context_limits import enforce_context_limits


def _build_msg(role: str, text: str):
    return {"role": role, "content": text}


class PoeContextLimitTestCase(unittest.TestCase):
    def test_enforce_context_limits_trims_old_messages(self):
        payload = {
            "messages": [
                _build_msg("system", "sys"),
                _build_msg("user", "old" * 2000),
                _build_msg("assistant", "ack"),
                _build_msg("user", "keep me"),
            ]
        }
        settings = Settings(context_window=200)
        updated, meta = enforce_context_limits(payload, settings, "poe:claude-opus-4.5")

        self.assertGreaterEqual(meta["dropped"], 1)
        self.assertEqual(updated["messages"][0]["role"], "system")
        self.assertEqual(updated["messages"][-1]["content"], "keep me")
        self.assertLessEqual(meta["after"], meta["budget"])

    def test_enforce_context_limits_truncates_system_when_alone_too_large(self):
        payload = {"messages": [_build_msg("system", "x" * 10000)]}
        settings = Settings(context_window=120)
        updated, meta = enforce_context_limits(payload, settings, "poe:claude-opus-4.5")

        self.assertEqual(len(updated["messages"]), 1)
        self.assertLess(len(updated["messages"][0]["content"]), 10000)
        self.assertLessEqual(meta["after"], meta["budget"])


if __name__ == "__main__":
    unittest.main()
