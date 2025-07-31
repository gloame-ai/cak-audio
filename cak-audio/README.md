# CAK: Emergent Audio Effects from Minimal Deep Learning

A neural audio effect learned from 200 samples using a single 3×3 convolutional kernel.

## Overview

CAK (Conditioning Aware Kernels) discovers audio transformations directly from data through adversarial training. Unlike traditional DSP effects with fixed behaviors, CAK learns an adaptive transformation that responds differently to different input characteristics.

Read the paper: 

Listen to demos of observed behavior [here](https://drive.google.com/drive/folders/1SRRxOFX6zX1jJoMMP-KKnqdN4D2U38O3?usp=drive_link).


Demo audio clips are provided for research and demonstration purposes only. Commercial use is not permitted. The clips remain property of their respective copyright holders and are used here under fair use.

## Installation

```bash
git clone https://github.com/gloame-ai/cak-audio.git
cd cak-audio
pip install -r requirements.txt
```
Note: The GUI requires tkinter, which usually comes with Python. If you get an import error:

Ubuntu/Debian: sudo apt-get install python3-tk
macOS: Should be included with Python
Windows: Should be included with Python

## Usage
Quick Start (GUI)
```bash
python cak_gui_minimal.py
```

Click "Drop audio file or browse" to load an audio file
Adjust the texture control slider (0 = no effect, 0.3+ = increasing effect)
Click "PROCESS" to apply the effect
Use playback controls to compare original vs processed
Save your processed audio with the "Save" button

Note: Processing time depends on audio length. A 2-minute file takes ~3-4 seconds on Apple Silicon.

Training Your Own Model
1. Prepare your dataset in the format shown in `dataset_format_example.py`
2. Generate STFT magnitude spectrograms using the parameters in `global_normalization_stats.json` (or adapt to your needs)
3. Adjust hyperparameters in the training script if needed
4. Run: `python cak_main_sandbox.py`

Note: Preprocessing code is not included, as implementations vary by use case. The expected format is magnitude spectrograms (2048 FFT, 512 hop) saved as .npz files with accompanying JSON metadata.

Training with 200 15-second samples with our configuration takes ~2 hours for 100 epochs on Apple M4 (48GB). We have found that the model generalizes meaningfully by epoch 75, it is worth experimenting with different checkpoints to see what your model has learned along the way.

## How It Works 
CAK uses a simple principle:
output = input + (learned_pattern × control)
The "audit game" (AuGAN) trains both generator and discriminator to cooperate in verifying that the control value was correctly applied, leading to learned transformations. Users should feel free to experiment with alternate kernel configurations or attempting to encode specific attributes paired with the control value. This is an area of research we are performing, with some promising results, but more ablations are needed. 

Like any audio effect, results vary by source material. Some audio will result in a more nuanced effect than others. We have found that transient heavy material (like percussion/drum loops) respond very well to this implementation of the CAK processor. Further, dense mid-range spectra with rich harmonic content appears to generate a temporal smearing effect, similar to what one may find in a chorus or phaser. We also acknowledge the limitations of mono outputs in the GUI. As we are introducing a baseline method, future applications will include stereo with further research. Happy experimenting! 

## Project Structure
cak-audio/
├── cak_gui_minimal.py           # GUI application
├── cak_main_sandbox.py          # Training script  
├── dataset_format_example.py    # Dataset structure example
├── wgan_grain_output/
│   └── final_wgan_grain.pt      # Pre-trained weights (694 KB)
├── examples/                       # Additional ablations
├── requirements.txt             # Python dependencies
└── README.md                    # This file

## Examples 
cak_detector_analysis.png - shown in paper, figure of learned kernel behavior
cak_freq_response_analysis.png - demonstrates learned frequency response of the kernel
extracted_training_history.png - shown in paper, training metrics
spectral_difference_validation.png - demonstrates that CAK processing produces spectral modifications beyond simple amplitude scaling, with frequency-dependent filtering effects visible in the spectral difference plot
test_signal_analysis.png - test signals (sine waves, noise, chirps, impulses) reveal CAK's frequency-dependent processing, showing adaptive transformation based on input characteristics.

## Troubleshooting
GUI appears cut off: Adjust window_width (line ~560) in cak_gui_minimal.py
Import errors: Make sure all requirements are installed, especially sounddevice for audio playback
CUDA/MPS errors: The code automatically falls back to CPU if GPU isn't available

## License
MIT License - see LICENSE file for details

## Authors
Austin Rockman (austin@gloame.ai)

## Acknowledgments
Roopam Garg, also of Gloame AI, implemented the demonstration GUI, contributed to identity preservation logic, and provided iterative feedback.
