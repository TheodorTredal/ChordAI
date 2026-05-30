import logging
from concurrent.futures import ThreadPoolExecutor
from svco.pulse_ingest import run_pulse_ingest_continuous
from svco.transcription_engine import process_turn
from svco.agent_orchestrator import orchestrate
from svco.output_synthesis import synthesize_output

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting continuous background pipeline. Ctrl+C to exit.")

    # Master state for the current full session
    session_speech = []
    session_chords = []

    # ThreadPool allows background processing while the mic keeps listening
    executor = ThreadPoolExecutor(max_workers=3)
    futures = []

    # Worker function to transcribe audio in the background
    def background_process(s_bytes, c_bytes):
        payload = process_turn(s_bytes, c_bytes, user_instruction="")
        if payload["user_speech"]:
            session_speech.append(payload["user_speech"])
        if payload["user_chords"]:
            session_chords.extend(payload["user_chords"])
        logging.info("Background sub-turn transcribed.")

    try:
        # Loop over the continuous audio generator
        for event_type, data in run_pulse_ingest_continuous(short_pause_s=1.2, long_pause_s=4.0):
            
            if event_type == "sub_turn":
                speech_bytes, chord_bytes = data
                logging.info(f"Sub-turn captured (speech: {len(speech_bytes)}, chord: {len(chord_bytes)}). Processing in background...")
                
                # Hand the audio off to the background thread; do not block!
                future = executor.submit(background_process, speech_bytes, chord_bytes)
                futures.append(future)

            elif event_type == "session_complete":
                logging.info("Long pause detected (4.0s). Finalizing payload...")
                
                # Wait for any lingering background transcription to finish
                for f in futures:
                    f.result()
                futures.clear()

                if not session_speech and not session_chords:
                    logging.info("Session was empty, skipping Claude.")
                    continue

                # Combine everything into one final payload
                final_payload = {
                    "user_speech": " ".join(session_speech),
                    "user_chords": list(session_chords), 
                    "user_instruction": " ".join(session_speech)
                }

                logging.info(f"Final Payload Assembled: {final_payload}")

                # Reset the master state for your next song/prompt
                session_speech.clear()
                session_chords.clear()

                # Step 3: Orchestrate agent interaction
                response_json = orchestrate(final_payload)
                logging.info(f"Agent response: {response_json}")

                # Step 4: Output synthesis
                synthesize_output(response_json)

    except KeyboardInterrupt:
        print("\nExiting pipeline.")
    except Exception as e:
        logging.exception(f"Error in main pipeline: {e}")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()