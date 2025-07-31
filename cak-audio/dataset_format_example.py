"""
Example dataset format for training CAK

The training expects:
1. A JSON file with metadata (a normalization stats or global preprocessing stats JSON is also helpful)
2. Pre-computed STFT magnitude spectrograms saved as .npz files - or whatever preprocessing schema one chooses
"""

# Note: training script currently expects 'grain_density' as the key but this example shows "effect_density" to
# make the core idea clear.

import json

example_json = {
    "segments": [
        {
            "stft_path": "spectrograms/sample_001.npz",
            "effect_density": 0.2  # control value
        },
        {
            "stft_path": "spectrograms/sample_002.npz",
            "effect_density": 0.5
        },
        # ... more segments
    ]
}

# Each .npz file should contain:
# - 'magnitude': the STFT magnitude array

print("Example dataset structure:")
print(json.dumps(example_json, indent=2))