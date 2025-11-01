# CAK: Emergent Audio Effects from Minimal Deep Learning

Generative neural audio effects learned from 200 samples using a single 3×3 convolutional kernel.

## Overview

CAK (Conditioning Aware Kernels) discovers audio transformations directly from data through adversarial training. Unlike traditional DSP effects with fixed behaviors, CAK learns an adaptive transformation that responds differently to different input characteristics.

Read the paper: [here](https://arxiv.org/abs/2508.02643) 

The trained model is easiest to use via Hugging Face, simply upload your sample (mono or stereo) and slide the "Texture Amount" to control the intensity of the neural transformation: https://huggingface.co/spaces/gloameai/cak-audio-processor

## Installation

```bash
git clone https://github.com/gloame-ai/cak-audio.git
cd cak-audio
pip install -r requirements.txt
```

Note: The GUI requires tkinter, which usually comes with Python on Mac & Windows. 

## Usage
Quick Start (GUI)
```bash
python cak_gui_minimal.py
```

1. Click "Drop audio file or browse" to load an audio file
2. Adjust the texture control slider (0 = no effect, 0.3+ = increasing effect)
3. Click "PROCESS" to apply the effect
4. Use playback controls to compare original vs processed
5. Save your processed audio with the "Save" button

Note: Processing time depends on audio length. A 2-minute file takes ~3-4 seconds on Apple Silicon.

Training Your Own Model
1. Prepare your dataset, drop files into a "samples" folder and run `generate_texture_samples.py` (applies random control value scalars across samples)
2. Generate STFT magnitude spectrograms with `dataset_preproc.py` (or adapt to your needs)
3. Adjust hyperparameters in the training script if needed
4. Run: `python cak_train` 

Note: Preprocessing code will truncate your samples to 15 seconds and apply a fade in/out to each clip. Please note the "self.gamma = 0.85" value. This is an optional midtone contrast boost applied during STFT normalization. This enhances mid-range spectral features and can be adjusted or removed based on your data characteristics. The training set was heavy in low end and gamma was used to offset potential spectral bias.
  
Training with 200 15-second samples with our configuration takes ~2 hours for 100 epochs on Apple M4 (48GB). We have found that the model generalizes meaningfully by epoch 75, it is worth experimenting with different checkpoints to see what your model has learned along the way. Given that this is a single kernel method, learning should be fairly rapid. 

## How It Works 
CAK uses a simple principle:
output = input + (learned_pattern × control)
The "audit game" (AuGAN) trains both generator and discriminator to cooperate in verifying that the control value was correctly applied, leading to learned transformations. Users should feel free to experiment with alternate kernel configurations or attempting to encode specific attributes paired with the control value.  

Like any audio effect, results vary by source material. Some audio will result in a more nuanced effect than others. We have found that transient heavy material (like percussion/drum loops) respond very well to this implementation of the CAK processor. Further, dense mid-range spectra with rich harmonic content appears to generate a smearing effect, similar to what one may find in a chorus or phaser. And at other times, the model can sound like a full blown comb filter.  

## Project Structure
```
cak-audio/
├── cak_gui_minimal.py           # GUI application
├── cak_train.py                 # Training script
├── dataset_preproc.py           # Audio preprocessing
├── generate_texture_samples     # Applies random control value scalars to each sample in dataset
├── examples/                    # Additional ablations
├── norm_stats/                  # Normalization parameters
│   └── global_normalization_stats.json
├── wgan_grain_output/           # Pre-trained model
│   └── final_wgan_grain.pt      # Weights (694 KB)
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
└── README.md
```                 

## Examples 
```
cak_detector_analysis.png - shown in paper, figure of learned kernel behavior
cak_freq_response_analysis.png - demonstrates learned frequency response of the kernel
extracted_training_history.png - shown in paper, training metrics
spectral_difference_validation.png - demonstrates that CAK processing produces spectral modifications
beyond simple amplitude scaling, with frequency-dependent filtering effects visible in the spectral difference plot
test_signal_analysis.png - test signals (sine waves, noise, chirps, impulses) reveal CAK's
frequency-dependent processing, showing adaptive transformation based on input characteristics.
```  

## Troubleshooting
```
GUI appears cut off: Adjust window_width (line ~560) in cak_gui_minimal.py
Import errors: Make sure all requirements are installed, especially sounddevice for audio playback
CUDA/MPS errors: The code automatically falls back to CPU if GPU isn't available
```

## License
MIT License - see LICENSE file for details

## Authors
Austin Rockman (austin@gloame.ai)

## Acknowledgments
Roopam Garg, also of Gloame AI, implemented the demonstration GUI, contributed to identity preservation logic, and provided iterative feedback.

## Citations
Rockman, A. (2025). CAK: Emergent Audio Effects from Minimal Deep Learning





