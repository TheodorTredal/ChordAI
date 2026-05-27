import unittest
from unittest.mock import patch

from svco.transcription_engine import build_payload, process_turn


class TranscriptionPayloadTests(unittest.TestCase):
    def test_build_payload_schema(self):
        payload = build_payload(
            user_speech="I was thinking of this intro",
            user_chords=["Cmaj7", "Am7", "Dm7", "G7"],
            user_instruction="Give me possible chords for a chorus.",
        )
        self.assertEqual(
            payload,
            {
                "user_speech": "I was thinking of this intro",
                "user_chords": ["Cmaj7", "Am7", "Dm7", "G7"],
                "user_instruction": "Give me possible chords for a chorus.",
            },
        )

    @patch("svco.transcription_engine.transcribe_chords", return_value=["Dm7", "G7"])
    @patch("svco.transcription_engine.transcribe_voice", return_value="Use this as intro")
    def test_process_turn_aggregates_components(self, *_):
        payload = process_turn(b"voice", b"chords", "Need a chorus")
        self.assertEqual(payload["user_speech"], "Use this as intro")
        self.assertEqual(payload["user_chords"], ["Dm7", "G7"])
        self.assertEqual(payload["user_instruction"], "Need a chorus")


if __name__ == "__main__":
    unittest.main()
