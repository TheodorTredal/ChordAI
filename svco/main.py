import logging
from stream_ingest import run_stream_ingest
from transcription_engine import transcribe
from agent_orchestrator import orchestrate
from output_synthesis import synthesize_output

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting pipeline (real mic input). Ctrl+C to exit.")

    while True:
        try:
            speech_bytes, chord_bytes = run_stream_ingest()
            logging.info(f"Buffer received (speech: {len(speech_bytes)} bytes, chord: {len(chord_bytes)} bytes).")

            # Step 2: Transcribe/parse buffers
            payload = transcribe(speech_bytes, chord_bytes)
            logging.info(f"Payload: {payload}")

            # Step 3: Orchestrate agent interaction (Claude/MCP stub or real)
            response_json = orchestrate(payload)
            logging.info(f"Agent response: {response_json}")

            # Step 4: Output synthesis (UI, TTS, etc.)
            synthesize_output(response_json)

        except KeyboardInterrupt:
            print("Exiting pipeline.")
            break
        except Exception as e:
            logging.exception(f"Error in main pipeline: {e}")

if __name__ == "__main__":
    main()