import unittest

from agent_orchestrator import parse_dual_channel_response
from output_synthesis import parse_response_json, process_agent_response


class DualChannelParsingTests(unittest.TestCase):
    def test_agent_parser_accepts_valid_schema(self):
        parsed = parse_dual_channel_response('{"speech_text":"Hello there","chord_data":["Cmaj7","Am7"]}')
        self.assertEqual(parsed["speech_text"], "Hello there")
        self.assertEqual(parsed["chord_data"], ["Cmaj7", "Am7"])

    def test_agent_parser_rejects_invalid_schema(self):
        with self.assertRaises(ValueError):
            parse_dual_channel_response('{"speech_text":42,"chord_data":["Cmaj7"]}')

    def test_output_processor_updates_ui_buffer(self):
        ui_buffer = ["old"]
        speech_text, chords = process_agent_response(
            '{"speech_text":"Try this progression","chord_data":["Fmaj7","Em7"]}',
            ui_chord_buffer=ui_buffer,
        )

        self.assertEqual(speech_text, "Try this progression")
        self.assertEqual(chords, ["Fmaj7", "Em7"])
        self.assertEqual(ui_buffer, ["Fmaj7", "Em7"])

    def test_output_parse_rejects_invalid_chord_data(self):
        with self.assertRaises(ValueError):
            parse_response_json('{"speech_text":"ok","chord_data":"Fmaj7"}')


if __name__ == "__main__":
    unittest.main()
