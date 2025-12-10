import logging
import unittest

from cc_adapter import streaming


class DummyResponse:
    def __init__(self):
        self.headers = {}
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}'
        raise ValueError("boom")

    def close(self):
        self.closed = True


class DummyHandler:
    def __init__(self):
        self.buffer = b""
        self.close_connection = False
        self.wfile = self

    def write(self, data: bytes):
        self.buffer += data

    def flush(self):
        return


class StreamingErrorHandlingTestCase(unittest.TestCase):
    def test_streaming_error_sends_stop_and_error_event(self):
        handler = DummyHandler()
        resp = DummyResponse()
        logger = logging.getLogger("stream-test")
        logger.setLevel(logging.CRITICAL)

        streaming.stream_openai_response(resp, "poe:deepseek-v3.2", {}, handler, logger)

        body = handler.buffer.decode("utf-8")
        self.assertTrue(handler.close_connection)
        self.assertIn("stop_reason\": \"error\"", body)
        self.assertIn("event: error", body)


if __name__ == "__main__":
    unittest.main()
