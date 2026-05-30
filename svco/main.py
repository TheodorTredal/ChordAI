import logging
from concurrent.futures import ThreadPoolExecutor
from svco.pulse_ingest import run_pulse_ingest_continuous
from svco.transcription_engine import process_turn
from svco.sender import send_to_server

logging.basicConfig(level=logging.INFO)


def main():
    logging.info("Starting continuous background pipeline. Ctrl+C to exit.")

    session_speech: list[str] = []
    session_chords: list = []

    executor = ThreadPoolExecutor(max_workers=3)
    futures = []

    def background_process(s_bytes: bytes, c_bytes: bytes) -> None:
        payload = process_turn(s_bytes, c_bytes, user_instruction="")
        if payload["user_speech"]:
            session_speech.append(payload["user_speech"])
        if payload["user_chords"]:
            session_chords.extend(payload["user_chords"])
        logging.info("Background sub-turn transcribed.")

    try:
        for event_type, data in run_pulse_ingest_continuous(short_pause_s=1.2, long_pause_s=4.0):

            if event_type == "sub_turn":
                speech_bytes, chord_bytes = data
                logging.info(
                    "Sub-turn captured (speech: %d bytes, chord: %d bytes). Processing in background...",
                    len(speech_bytes), len(chord_bytes),
                )
                futures.append(executor.submit(background_process, speech_bytes, chord_bytes))

            elif event_type == "session_complete":
                logging.info("Long pause detected — finalizing session.")

                # Wait for all background transcription to finish before sending
                for f in futures:
                    f.result()
                futures.clear()

                if not session_speech and not session_chords:
                    logging.info("Session was empty, skipping.")
                    continue

                combined_speech = " ".join(session_speech).strip()
                combined_chords = list(session_chords)

                logging.info(
                    "Sending to ChordAI server: speech=%r chords=%d segments",
                    combined_speech[:80], len(combined_chords),
                )

                session_speech.clear()
                session_chords.clear()

                result = send_to_server(combined_speech, combined_chords)

                if result:
                    logging.info(
                        "SongResult received: genre=%s decade=%s lyrics_len=%d",
                        result.get("genre"), result.get("decade"),
                        len(result.get("lyrics", "")),
                    )
                else:
                    logging.warning("Server returned no result — is the Go server running?")

    except KeyboardInterrupt:
        print("\nExiting pipeline.")
    except Exception:
        logging.exception("Error in main pipeline")
    finally:
        executor.shutdown(wait=False)


if __name__ == "__main__":
    main()