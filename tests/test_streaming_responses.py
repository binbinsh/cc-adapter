import logging
import unittest

from cc_adapter import streaming


class DummyResponse:
    def __init__(self, lines):
        self._lines = list(lines)
        self.headers = {}
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

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


class ResponsesStreamingBridgeTestCase(unittest.TestCase):
    def test_streaming_text_delta(self):
        handler = DummyHandler()
        resp = DummyResponse(
            [
                b'data: {"type":"response.created","response_id":"resp_1"}',
                b'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}',
                b'data: {"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":2,"output_tokens":1}}}',
            ]
        )
        logger = logging.getLogger("responses-stream-test")
        logger.setLevel(logging.CRITICAL)

        streaming.stream_responses_response(resp, "codex:gpt-5.1-codex", {}, handler, logger)

        body = handler.buffer.decode("utf-8")
        self.assertIn("event: message_start", body)
        self.assertIn("text_delta", body)
        self.assertIn("hello", body)
        self.assertIn("\"stop_reason\": \"end_turn\"", body)
        self.assertIn("\"input_tokens\": 2", body)

    def test_streaming_tool_call_arguments(self):
        handler = DummyHandler()
        resp = DummyResponse(
            [
                b'data: {"type":"response.created","response_id":"resp_2"}',
                b'data: {"type":"response.output_item.added","response_id":"resp_2","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"do_it","arguments":""}}',
                b'data: {"type":"response.function_call_arguments.delta","response_id":"resp_2","item_id":"fc_1","output_index":0,"delta":"{\\\"x\\\":1"}',
                b'data: {"type":"response.function_call_arguments.delta","response_id":"resp_2","item_id":"fc_1","output_index":0,"delta":"}"}',
                b'data: {"type":"response.completed","response":{"id":"resp_2","usage":{"input_tokens":3,"output_tokens":1}}}',
            ]
        )
        logger = logging.getLogger("responses-tool-test")
        logger.setLevel(logging.CRITICAL)

        streaming.stream_responses_response(resp, "codex:gpt-5.1-codex", {}, handler, logger)

        body = handler.buffer.decode("utf-8")
        self.assertIn("\"type\": \"tool_use\"", body)
        self.assertIn("\"id\": \"call_1\"", body)
        self.assertIn("\"name\": \"do_it\"", body)
        self.assertIn("input_json_delta", body)
        self.assertIn("\"stop_reason\": \"tool_use\"", body)


if __name__ == "__main__":
    unittest.main()

