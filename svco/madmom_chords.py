"""Deep learning offline chord recognition using Madmom."""

import logging
import numpy as np

try:
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
except ImportError as e:
    raise ImportError("Madmom is not installed. Please install it to use this feature.") from e

def transcribe_chords_madmom(file_path: str) -> list:
    """
    Ingests an audio file path, runs CNN feature extraction, 
    and applies CRF temporal smoothing to output recognized chords.
    """
    logging.info(f"Running Madmom deep learning extraction on: {file_path}")
    
    try:
        # Step 1: The CNN analyzes the audio and extracts raw chord probabilities
        feat_processor = CNNChordFeatureProcessor()
        features = feat_processor(file_path)

        # Step 2: The CRF smooths the probabilities into definitive chord segments
        decode_processor = CRFChordRecognitionProcessor()
        segments = decode_processor(features)
        
        # Madmom outputs a numpy array of [start_time, end_time, chord_label]
        # We only want to extract the unique, sequential labels for the UI
        filtered_chords = []
        for segment in segments:
            # segment[2] is the chord label string
            label = str(segment[2]) 
            
            if label != "N":  # Ignore "No Chord" segments
                if not filtered_chords or filtered_chords[-1] != label:
                    filtered_chords.append(label)
                    
        return filtered_chords

    except Exception as e:
        logging.error(f"Madmom processing failed: {e}")
        return []