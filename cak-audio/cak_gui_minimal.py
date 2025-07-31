"""
CAK Neural Audio Processor GUI

This provides an interactive interface for the CAK (Conditioning Aware Kernels) audio effect.
Load any audio file and adjust the texture control to apply the learned neural transformation.

It's best to think about this effect like any other DSP effect, on some sounds it will be compelling, others
it may not, but of course that's subjective.

Note: If the GUI appears truncated on your display, you may need to:
- Adjust the window_width value (line ~350)
- Modify the padding values in main_container (line ~365)
- Scale the font sizes in MinimalDarkStyle class

Note: The GUI accepts various formats (WAV, MP3, FLAC, M4A), we recommend using 44.1kHz 16-bit WAV files for
best quality and to avoid resampling artifacts. Other formats (not .aif) will be automatically converted but 
may introduce quality loss or errors.

Author: Austin Rockman (austin@gloame.ai) & Roopam Garg (roopam@gloame.ai)
Date: July 2025
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
from pathlib import Path
import json
import sys

sys.path.append('.')

# cak model imports
from cak_main_sandbox import SharedTextureDetector, TextureGenerator


class MinimalDarkStyle:
    """minimal dark theme matching the sleek aesthetic"""

    # pure black/dark grays
    BG_PRIMARY = "#000000"
    BG_SECONDARY = "#0a0a0a"
    BG_CARD = "#141414"
    BG_HOVER = "#1a1a1a"

    # text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#888888"
    TEXT_MUTED = "#555555"

    # accent
    ACCENT = "#ffffff"

    # fonts
    FONT_TITLE = ("SF Pro Display", 32, "normal")
    FONT_SUBTITLE = ("SF Pro Display", 14, "normal")
    FONT_BODY = ("SF Pro Display", 12, "normal")
    FONT_SMALL = ("SF Pro Display", 10, "normal")
    FONT_MONO = ("SF Mono", 10, "normal")

    @classmethod
    def apply_style(cls, root):
        """apply minimal dark theme"""
        style = ttk.Style(root)

        # configure root
        root.configure(bg=cls.BG_PRIMARY)

        # use default theme as base
        style.theme_use('default')

        # general configuration
        style.configure(".",
                        background=cls.BG_PRIMARY,
                        foreground=cls.TEXT_PRIMARY,
                        borderwidth=0,
                        focuscolor="none",
                        highlightthickness=0,
                        relief="flat")

        # frame styles
        style.configure("TFrame", background=cls.BG_PRIMARY, borderwidth=0)
        style.configure("Card.TFrame", background=cls.BG_CARD, relief="flat", borderwidth=1)

        # label styles
        style.configure("TLabel", background=cls.BG_PRIMARY, foreground=cls.TEXT_PRIMARY)
        style.configure("Title.TLabel", font=cls.FONT_TITLE, background=cls.BG_PRIMARY)
        style.configure("Subtitle.TLabel", font=cls.FONT_SUBTITLE, foreground=cls.TEXT_SECONDARY)
        style.configure("Body.TLabel", font=cls.FONT_BODY)
        style.configure("Small.TLabel", font=cls.FONT_SMALL, foreground=cls.TEXT_SECONDARY)
        style.configure("Mono.TLabel", font=cls.FONT_MONO, foreground=cls.TEXT_MUTED)

        # minimal button style
        style.configure("Minimal.TButton",
                        background=cls.BG_PRIMARY,
                        foreground=cls.TEXT_PRIMARY,
                        borderwidth=1,
                        focuscolor="none",
                        highlightthickness=0,
                        padding=(30, 12),
                        relief="flat",
                        bordercolor=cls.TEXT_MUTED,
                        lightcolor=cls.TEXT_MUTED,
                        darkcolor=cls.TEXT_MUTED,
                        font=cls.FONT_BODY)

        style.map("Minimal.TButton",
                  background=[("active", cls.BG_HOVER)],
                  bordercolor=[("active", cls.TEXT_PRIMARY)],
                  foreground=[("disabled", cls.TEXT_MUTED)])

        # file display style
        style.configure("File.TFrame",
                        background=cls.BG_CARD,
                        borderwidth=1,
                        relief="flat",
                        bordercolor=cls.TEXT_MUTED)

        # scale style - minimal
        style.configure("Minimal.Horizontal.TScale",
                        background=cls.BG_PRIMARY,
                        troughcolor=cls.TEXT_MUTED,
                        borderwidth=0,
                        lightcolor=cls.BG_PRIMARY,
                        darkcolor=cls.BG_PRIMARY,
                        sliderwidth=12,
                        sliderlength=12)

        style.map("Minimal.Horizontal.TScale",
                  troughcolor=[("active", cls.TEXT_PRIMARY)])

        # progress bar - minimal line
        style.configure("Minimal.Horizontal.TProgressbar",
                        background=cls.TEXT_PRIMARY,
                        troughcolor=cls.TEXT_MUTED,
                        borderwidth=0,
                        lightcolor=cls.TEXT_PRIMARY,
                        darkcolor=cls.TEXT_PRIMARY,
                        thickness=2)

        # configure matplotlib
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': cls.BG_PRIMARY,
            'axes.facecolor': cls.BG_PRIMARY,
            'axes.edgecolor': cls.TEXT_MUTED,
            'axes.labelcolor': cls.TEXT_SECONDARY,
            'text.color': cls.TEXT_PRIMARY,
            'xtick.color': cls.TEXT_MUTED,
            'ytick.color': cls.TEXT_MUTED,
            'grid.color': cls.BG_CARD,
            'grid.alpha': 0.3,
        })


class CircularSlider(tk.Canvas):
    """custom circular slider widget for aesthetic control"""

    def __init__(self, parent, from_=0.0, to=2.0, value=0.0, command=None, **kwargs):
        super().__init__(parent, height=20, bg=MinimalDarkStyle.BG_PRIMARY,
                         highlightthickness=0, cursor="pointinghand", **kwargs)

        self.from_ = from_
        self.to = to
        self.value = value
        self.command = command

        # slider dimensions
        self.thumb_radius = 8
        self.line_height = 2
        self.padding = self.thumb_radius + 2

        # create slider elements
        self.line = None
        self.thumb = None
        self.dragging = False

        # bind events
        self.bind("<Configure>", self._on_configure)
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_configure(self, event):
        """redraw slider when window is resized"""
        self.delete("all")

        # get canvas dimensions
        w = self.winfo_width()
        h = self.winfo_height()

        if w <= 1:  # canvas not ready yet
            return

        # draw the track line
        y = h // 2
        self.line = self.create_rectangle(
            self.padding, y - self.line_height // 2,
                          w - self.padding, y + self.line_height // 2,
            fill=MinimalDarkStyle.TEXT_MUTED,
            outline=""
        )

        # draw the thumb circle
        x = self._value_to_x(self.value)
        self.thumb = self.create_oval(
            x - self.thumb_radius, y - self.thumb_radius,
            x + self.thumb_radius, y + self.thumb_radius,
            fill=MinimalDarkStyle.TEXT_PRIMARY,
            outline="",
            tags="thumb"
        )

    def _value_to_x(self, value):
        """convert value to x coordinate"""
        w = self.winfo_width()
        if w <= 1:
            return self.padding

        # map value to x position
        ratio = (value - self.from_) / (self.to - self.from_)
        x = self.padding + ratio * (w - 2 * self.padding)
        return max(self.padding, min(x, w - self.padding))

    def _x_to_value(self, x):
        """convert x coordinate to value"""
        w = self.winfo_width()
        if w <= 2 * self.padding:
            return self.from_

        # map x position to value
        ratio = (x - self.padding) / (w - 2 * self.padding)
        ratio = max(0, min(1, ratio))
        return self.from_ + ratio * (self.to - self.from_)

    def _on_click(self, event):
        """handle mouse click"""
        # check if clicked on thumb
        bbox = self.bbox("thumb")
        if bbox and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
            self.dragging = True
            self.itemconfig("thumb", fill=MinimalDarkStyle.TEXT_SECONDARY)
        else:
            # jump to clicked position
            self._update_value(event.x)

    def _on_drag(self, event):
        """handle mouse drag"""
        if self.dragging:
            self._update_value(event.x)

    def _on_release(self, event):
        """handle mouse release"""
        self.dragging = False
        self.itemconfig("thumb", fill=MinimalDarkStyle.TEXT_PRIMARY)

    def _update_value(self, x):
        """update slider value based on x position"""
        new_value = self._x_to_value(x)
        if new_value != self.value:
            self.value = new_value

            # update thumb position
            thumb_x = self._value_to_x(self.value)
            y = self.winfo_height() // 2
            self.coords(
                self.thumb,
                thumb_x - self.thumb_radius, y - self.thumb_radius,
                thumb_x + self.thumb_radius, y + self.thumb_radius
            )

            # call command callback
            if self.command:
                self.command(self.value)

    def get(self):
        """get current value"""
        return self.value

    def set(self, value):
        """set slider value programmatically"""
        self.value = max(self.from_, min(value, self.to))
        self._on_configure(None)


class AudioProcessor:
    """audio processor for CAK using iSTFT"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # audio params stats from training
        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_size = 512
        self.win_size = 2048

        # global norm stats from training
        stats_path = 'norm_stats/global_normalization_stats.json'
        with open(stats_path, 'r') as f:
            self.global_stats = json.load(f)

        print(f"Loaded global stats: low={self.global_stats['global_low']:.3f}, "
              f"high={self.global_stats['global_high']:.3f}")

        # create model architecture
        self.shared_detector = SharedTextureDetector()
        self.generator = TextureGenerator(self.shared_detector)

        # load trained weights
        checkpoint_path = 'wgan_grain_output/final_wgan_grain.pt'
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(checkpoint['generator'])
            self.shared_detector.load_state_dict(checkpoint['shared_detector'])

            # set epoch for temperature (should be at final value, if last checkpoint is used)
            if 'epoch' in checkpoint:
                self.generator.update_epoch(checkpoint['epoch'])

            print(f"Model loaded from: {checkpoint_path}")
            print(f"Scale parameter: {self.generator.cak.scale.item():.3f}")
            print(f"Temperature: {self.generator.cak.temperature:.1f}")
            print(f"Threshold: {self.generator.cak.threshold.item():.2f}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        # move to device and eval
        self.shared_detector.to(self.device)
        self.generator.to(self.device)
        self.generator.eval()

        # sanity check to make sure iSTFT is calibrated
        print("Using iSTFT for phase-preserving reconstruction")

    def normalize_stft(self, stft_mag):
        """stft norm that matches training"""
        # sqrt compression
        sqrt_mag = np.sqrt(stft_mag)

        # log compression
        log_mag = np.log1p(sqrt_mag * self.global_stats['alpha'])

        # norm with global stats
        normalized = np.clip(
            (log_mag - self.global_stats['global_low']) /
            (self.global_stats['global_high'] - self.global_stats['global_low'] + 1e-8),
            0, 1
        )

        # gamma correction
        gamma_corrected = np.power(normalized, self.global_stats['gamma'])

        return gamma_corrected

    def denormalize_stft(self, normalized):
        """reverse the stft norm"""
        # ensure input is in valid range [0, 1]
        normalized = np.clip(normalized, 0, 1)

        # reverse gamma
        pre_gamma = np.power(normalized + 1e-8, 1.0 / self.global_stats['gamma'])

        # reverse norm
        log_mag = pre_gamma * (self.global_stats['global_high'] - self.global_stats['global_low']) + \
                  self.global_stats['global_low']

        # reverse log compression
        sqrt_mag = (np.exp(log_mag) - 1) / self.global_stats['alpha']
        sqrt_mag = np.maximum(sqrt_mag, 0)  # ensure non-negative

        # reverse sqrt compression
        mag = sqrt_mag ** 2

        # safety check
        mag = np.maximum(mag, 0)

        return mag

    def process_audio(self, audio_path, texture_amount, progress_callback=None):
        """process audio file with learned neuron"""

        # load audio
        if progress_callback:
            progress_callback(0.1, "Loading audio...")

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # process in chunks (15 seconds, matching training, chunks will take longer if more than 15 seconds)
        chunk_duration = 15.0
        chunk_samples = int(chunk_duration * self.sample_rate)
        hop_samples = chunk_samples // 2  # 50% overlap

        chunks = []
        for start in range(0, len(y) - chunk_samples + hop_samples, hop_samples):
            chunk = y[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                # pad last chunk
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            chunks.append((chunk, start))

        # if audio is shorter than chunk_samples, process as single chunk
        if len(chunks) == 0:
            chunk = np.pad(y, (0, max(0, chunk_samples - len(y))), mode='constant')
            chunks = [(chunk, 0)]

            print(f"\n=== Audio Info ===")
            print(f"Audio length: {len(y) / self.sample_rate:.2f} seconds")
            print(f"Chunk duration: {chunk_duration} seconds")
            print(f"Number of chunks: {len(chunks)}")
            print(f"Texture amount: {texture_amount}")

        processed_chunks = []
        total_chunks = len(chunks)

        for idx, (chunk, start_pos) in enumerate(chunks):
            if progress_callback:
                progress = 0.1 + (0.7 * idx / total_chunks)
                progress_callback(progress, f"Processing chunk {idx + 1}/{total_chunks}")

            # compute complex stft
            stft_complex = librosa.stft(
                chunk,
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size
            )

            # get magnitude and phase
            magnitude = np.abs(stft_complex)
            phase = np.angle(stft_complex)

            # norm magnitude
            normalized_mag = self.normalize_stft(magnitude)

            # process with trained neural network
            with torch.no_grad():
                # prepare 4d input [B, 1, F, T]
                mag_tensor = torch.FloatTensor(normalized_mag).unsqueeze(0).unsqueeze(0).to(self.device)
                texture_tensor = torch.FloatTensor([[texture_amount]]).to(self.device)

                # apply cak processing
                processed_mag, patterns = self.generator(mag_tensor, texture_tensor)
                processed_mag = processed_mag.cpu().numpy()[0, 0]

                input_mean = normalized_mag.mean()
                output_mean = processed_mag.mean()
                print(f"\nChunk {idx + 1}:")
                print(f"  Input mean: {input_mean:.6f}")
                print(f"  Output mean: {output_mean:.6f}")
                print(f"  Difference: {abs(output_mean - input_mean):.6f}")
                print(f"  Gate value: {self.generator.cak.soft_gate(texture_tensor).item():.3f}")

            # denorm back to linear magnitude
            processed_mag_linear = self.denormalize_stft(processed_mag)

            # reconstruct complex stft using original phase
            processed_stft = processed_mag_linear * np.exp(1j * phase)

            # convert back to audio using istft
            processed_chunk = librosa.istft(
                processed_stft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                length=chunk_samples
            )

            # sanity debug for first chunk
            if idx == 0:
                print(f"\nProcessing stats:")
                print(f"  Original magnitude range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
                print(f"  Normalized range: [{normalized_mag.min():.3f}, {normalized_mag.max():.3f}]")
                print(
                    f"  Processed magnitude range: [{processed_mag_linear.min():.3f}, {processed_mag_linear.max():.3f}]")
                print(f"  Texture amount: {texture_amount:.2f}")
                if abs(texture_amount) < 1e-6:
                    # check identity preservation
                    mag_diff = np.abs(processed_mag_linear - magnitude).mean()
                    print(f"  Identity check - mean magnitude difference: {mag_diff:.6f}")

            processed_chunks.append((processed_chunk, start_pos))

        # crossfade chunks
        if progress_callback:
            progress_callback(0.9, "Combining audio...")

        output_audio = self.combine_chunks(processed_chunks, len(y))

        # match rms to original
        original_rms = np.sqrt(np.mean(y ** 2))
        output_rms = np.sqrt(np.mean(output_audio ** 2))

        if output_rms > 0:
            rms_ratio = original_rms / output_rms
            rms_ratio = np.clip(rms_ratio, 0.1, 10.0)
            output_audio *= rms_ratio

        # soft clipping to prevent any peaks
        output_audio = np.tanh(output_audio * 0.95) / 0.95

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return output_audio, self.sample_rate

    def combine_chunks(self, chunks, target_length):
        """combine overlapping chunks with crossfade"""
        output = np.zeros(target_length)
        weights = np.zeros(target_length)

        for chunk, start_pos in chunks:
            # ensure 1d
            if chunk.ndim > 1:
                chunk = chunk.flatten()

            end_pos = min(start_pos + len(chunk), target_length)
            chunk_len = end_pos - start_pos

            # fade window
            window = np.ones(chunk_len)
            fade_len = min(self.hop_size // 2, chunk_len // 4)

            if start_pos > 0 and fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                window[:fade_len] = fade_in

            if end_pos < target_length and fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len)
                window[-fade_len:] = fade_out

            output[start_pos:end_pos] += chunk[:chunk_len] * window
            weights[start_pos:end_pos] += window

        # norm by weights
        mask = weights > 0
        output[mask] /= weights[mask]

        return output


class NeuralAudioProcessorGUI:
    """minimal dark gui for cak neural audio processor"""

    def __init__(self, root):
        self.root = root
        self.root.title("CAK Neural Processor")

        # get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # set window to 900px wide and full screen height
        window_width = 1000
        window_height = screen_height

        # center the window horizontally
        x = (screen_width - window_width) // 2
        y = 0

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(700, 950)
        self.root.resizable(True, True)

        # apply minimal dark theme
        MinimalDarkStyle.apply_style(self.root)

        # initialize processor
        try:
            self.processor = AudioProcessor()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize processor: {e}")
            root.destroy()
            return

        # audio data
        self.original_audio = None
        self.processed_audio = None
        self.sr = None
        self.current_file = None

        # processing queue
        self.process_queue = queue.Queue()

        # create gui
        self.create_widgets()

        # start processing
        self.processing = False
        self.check_queue()

    def create_widgets(self):
        """create minimal dark gui"""

        # main container - centered layout
        main_container = ttk.Frame(self.root, style="TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=60, pady=30)

        # title section
        title_label = ttk.Label(main_container, text="CAK Neural Processor",
                                style="Title.TLabel")
        title_label.pack(pady=(0, 5))

        subtitle_label = ttk.Label(main_container, text="Conditioning Aware Kernels",
                                   style="Subtitle.TLabel")
        subtitle_label.pack(pady=(0, 5))

        # file section
        file_container = ttk.Frame(main_container, style="File.TFrame", height=60, cursor="pointinghand")
        file_container.pack(fill=tk.X, pady=(0, 30))
        file_container.pack_propagate(False)

        file_inner = ttk.Frame(file_container, style="TFrame")
        file_inner.place(relx=0.5, rely=0.5, anchor="center")

        self.file_label = ttk.Label(file_inner, text="Drop audio file or browse",
                                    style="Body.TLabel", cursor="pointinghand")
        self.file_label.pack(side=tk.LEFT, padx=(20, 10))

        self.file_size_label = ttk.Label(file_inner, text="", style="Small.TLabel")
        self.file_size_label.pack(side=tk.LEFT, padx=(0, 20))

        # make file container clickable
        file_container.bind("<Button-1>", lambda e: self.browse_file())
        self.file_label.bind("<Button-1>", lambda e: self.browse_file())

        # amount control
        amount_label = ttk.Label(main_container, text="AMOUNT", style="Small.TLabel")
        amount_label.pack(anchor="w", pady=(0, 10))

        # slider container
        slider_frame = ttk.Frame(main_container, style="TFrame")
        slider_frame.pack(fill=tk.X, pady=(0, 5))

        # use custom circular slider
        self.texture_slider = CircularSlider(slider_frame, from_=0.0, to=2.0,
                                             value=0.0, command=self.update_texture_label)
        self.texture_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # value display
        value_frame = ttk.Frame(main_container, style="TFrame")
        value_frame.pack(fill=tk.X, pady=(0, 20))

        self.texture_label = ttk.Label(value_frame, text="0.00", style="Body.TLabel")
        self.texture_label.pack(side=tk.LEFT)

        self.gate_label = ttk.Label(value_frame, text="Gate: 0.002", style="Small.TLabel")
        self.gate_label.pack(side=tk.RIGHT)

        # info text box
        info_frame = ttk.Frame(main_container, style="Card.TFrame")
        info_frame.pack(fill=tk.X, pady=(0, 30))

        info_inner = ttk.Frame(info_frame, style="Card.TFrame")
        info_inner.pack(padx=20, pady=15)

        threshold = self.processor.generator.cak.threshold.item()
        info_lines = [
            f"Values < {threshold:.2f}: Minimal effect",
            f"Values ≥ {threshold:.2f}: Texture modulation active",
            "Zero: Identity preservation"
        ]

        for line in info_lines:
            line_label = ttk.Label(info_inner, text=line, style="Mono.TLabel")
            line_label.pack(anchor="w", pady=2)

        # process button
        self.process_button = ttk.Button(main_container, text="PROCESS",
                                         command=self.process_audio,
                                         state=tk.DISABLED,
                                         style="Minimal.TButton",
                                         cursor="pointinghand")
        self.process_button.pack(pady=(0, 30))

        # progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_container,
                                            variable=self.progress_var,
                                            maximum=1.0,
                                            style="Minimal.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(0, 2))

        self.status_label = ttk.Label(main_container, text="", style="Mono.TLabel")
        self.status_label.pack(pady=(0, 5))

        # visualization labels
        viz_label_frame = ttk.Frame(main_container, style="TFrame")
        viz_label_frame.pack(fill=tk.X, pady=(0, 5))

        # create two labels positioned at the same spots as the plots
        original_label = ttk.Label(viz_label_frame, text="ORIGINAL", style="Small.TLabel")
        original_label.pack(side=tk.LEFT, padx=(0, 0))

        # add spacer in middle to push processed label to the right
        spacer = ttk.Frame(viz_label_frame, style="TFrame")
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        processed_label = ttk.Label(viz_label_frame, text="PROCESSED", style="Small.TLabel")
        processed_label.pack(side=tk.LEFT, padx=(0, 0))

        # create figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 1))
        self.fig.patch.set_facecolor(MinimalDarkStyle.BG_PRIMARY)

        # configure axes - completely minimal
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(MinimalDarkStyle.BG_PRIMARY)
            # hide all spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            # remove all ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        self.fig.tight_layout(pad=0.5)

        canvas_frame = ttk.Frame(main_container, style="TFrame")
        canvas_frame.pack(fill=tk.X, pady=(0, 5))

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.X)

        # playback controls
        controls_frame = ttk.Frame(main_container, style="TFrame")
        controls_frame.pack(fill=tk.X)

        button_frame = ttk.Frame(controls_frame, style="TFrame")
        button_frame.pack()

        self.play_original_btn = ttk.Button(button_frame, text="Play Original",
                                            command=self.play_original,
                                            style="Minimal.TButton")
        self.play_original_btn.pack(side=tk.LEFT, padx=5)

        self.play_processed_btn = ttk.Button(button_frame, text="Play Processed",
                                             command=self.play_processed,
                                             style="Minimal.TButton",
                                             state=tk.DISABLED)
        self.play_processed_btn.pack(side=tk.LEFT, padx=5)

        self.ab_test_btn = ttk.Button(button_frame, text="A/B Test",
                                      command=self.ab_test,
                                      style="Minimal.TButton",
                                      state=tk.DISABLED)
        self.ab_test_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(button_frame, text="Save",
                                   command=self.save_processed,
                                   style="Minimal.TButton",
                                   state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop",
                                   command=self.stop_audio,
                                   style="Minimal.TButton")
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def update_texture_label(self, value):
        """update texture amount and gate display"""
        texture_val = float(value)
        self.texture_label.config(text=f"{texture_val:.2f}")

        # calculate gate value
        with torch.no_grad():
            texture_tensor = torch.tensor([[texture_val]]).to(self.processor.device)
            gate_value = self.processor.generator.cak.soft_gate(texture_tensor).item()
            self.gate_label.config(text=f"Gate: {gate_value:.3f}")

    def browse_file(self):
        """browse for audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")]
        )
        if filename:
            self.current_file = filename
            self.file_label.config(text=Path(filename).name)

            # get file size
            file_size = Path(filename).stat().st_size / (1024 * 1024)  # mb
            self.file_size_label.config(text=f"{file_size:.2f} MB")

            # enable process button
            self.process_button.config(state=tk.NORMAL, text="PROCESS")

            # load audio
            self.status_label.config(text=f"Loading audio...")
            self.original_audio, self.sr = librosa.load(filename, sr=self.processor.sample_rate, mono=True)
            duration = len(self.original_audio) / self.sr
            self.status_label.config(text=f"Loaded: {duration:.2f}s @ {self.sr}Hz")

            self.visualize_original()

    def visualize_original(self):
        """visualize original audio waveform"""
        if self.original_audio is None:
            return

        # clear and plot waveform
        self.ax1.clear()

        # downsample for visualization
        downsample = max(1, len(self.original_audio) // 5000)
        y_vis = self.original_audio[::downsample]
        x_vis = np.arange(len(y_vis)) * downsample / self.sr

        self.ax1.plot(x_vis, y_vis, color=MinimalDarkStyle.TEXT_SECONDARY, linewidth=0.5)
        self.ax1.set_xlim(0, len(self.original_audio) / self.sr)
        self.ax1.set_ylim(-1.1, 1.1)

        # remove all axes elements
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax1.tick_params(bottom=False, left=False)
        for spine in self.ax1.spines.values():
            spine.set_visible(False)

        self.canvas.draw()

    def process_audio(self):
        """process audio in background thread"""
        if self.processing:
            return

        self.processing = True
        self.process_button.config(state=tk.DISABLED, text="PROCESSING...")
        self.play_processed_btn.config(state=tk.DISABLED)
        self.ab_test_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._process_thread)
        thread.start()

    def _process_thread(self):
        """background processing thread"""
        try:
            texture_amount = self.texture_slider.get()

            self.processed_audio, self.sr = self.processor.process_audio(
                self.current_file,
                texture_amount,
                progress_callback=self._update_progress
            )

            self.process_queue.put(('complete', None))

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.process_queue.put(('error', error_msg))

    def _update_progress(self, progress, message):
        self.process_queue.put(('progress', (progress, message)))

    def check_queue(self):
        """check processing queue for updates"""
        try:
            while True:
                msg_type, data = self.process_queue.get_nowait()

                if msg_type == 'progress':
                    progress, message = data
                    self.progress_var.set(progress)
                    self.status_label.config(text=message)

                elif msg_type == 'complete':
                    self.visualize_processed()
                    self.processing = False
                    self.process_button.config(state=tk.NORMAL, text="PROCESS")
                    self.play_processed_btn.config(state=tk.NORMAL)
                    self.ab_test_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                    self.status_label.config(text="Processing complete")

                elif msg_type == 'error':
                    messagebox.showerror("Processing Error", data)
                    self.processing = False
                    self.process_button.config(state=tk.NORMAL, text="PROCESS")
                    self.status_label.config(text="Error occurred")

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def visualize_processed(self):
        """visualize processed audio waveform"""
        if self.processed_audio is None:
            return

        # clear and plot waveform
        self.ax2.clear()

        # downsample for visualization
        downsample = max(1, len(self.processed_audio) // 5000)
        y_vis = self.processed_audio[::downsample]
        x_vis = np.arange(len(y_vis)) * downsample / self.sr

        self.ax2.plot(x_vis, y_vis, color=MinimalDarkStyle.TEXT_PRIMARY, linewidth=0.5)
        self.ax2.set_xlim(0, len(self.processed_audio) / self.sr)
        self.ax2.set_ylim(-1.1, 1.1)

        # remove all axes elements
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.tick_params(bottom=False, left=False)
        for spine in self.ax2.spines.values():
            spine.set_visible(False)

        self.canvas.draw()

    def play_original(self):
        """play original audio"""
        if self.original_audio is None:
            return

        try:
            import sounddevice as sd
            sd.play(self.original_audio, self.sr)
        except ImportError:
            messagebox.showinfo("Info", "Install sounddevice: pip install sounddevice")

    def play_processed(self):
        """play processed audio"""
        if self.processed_audio is None:
            return

        try:
            import sounddevice as sd
            sd.play(self.processed_audio, self.sr)
        except ImportError:
            messagebox.showinfo("Info", "Install sounddevice: pip install sounddevice")

    def save_processed(self):
        """save processed audio"""
        if self.processed_audio is None:
            return

        filename = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )

        if filename:
            audio_clipped = np.clip(self.processed_audio, -1.0, 1.0)
            sf.write(filename, audio_clipped, self.sr)
            self.status_label.config(text=f"Saved: {Path(filename).name}")

    def stop_audio(self):
        """stop audio playback"""
        try:
            import sounddevice as sd
            sd.stop()
            self.status_label.config(text="Playback stopped")
        except ImportError:
            pass

    def ab_test(self):
        """a/b test between original and processed"""
        if self.original_audio is None or self.processed_audio is None:
            return

        try:
            import sounddevice as sd
            import time

            # play 2 seconds of each
            self.status_label.config(text="A/B Test: Original...")
            sd.play(self.original_audio[:self.sr * 2], self.sr)
            sd.wait()
            time.sleep(0.5)

            self.status_label.config(text="A/B Test: Processed...")
            sd.play(self.processed_audio[:self.sr * 2], self.sr)
            sd.wait()
            time.sleep(0.5)

            self.status_label.config(text="A/B Test: Original...")
            sd.play(self.original_audio[:self.sr * 2], self.sr)
            sd.wait()

            self.status_label.config(text="A/B Test complete")

        except ImportError:
            messagebox.showinfo("Info", "Install sounddevice: pip install sounddevice")


def main():
    """run the application"""
    root = tk.Tk()
    app = NeuralAudioProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()