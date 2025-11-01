import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random


class CAKDatasetBuilder:

    def __init__(self,
                 samples_root='./samples',
                 output_dir='./cak_dataset',
                 segment_duration=15.0,
                 fade_duration=2.0):

        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_size = 512
        self.win_size = 2048

        self.n_mels = 128
        self.fmin = 0
        self.fmax = None

        self.alpha = 10.0
        self.gamma = 0.85

        self.segment_duration = segment_duration
        self.fade_duration = fade_duration
        self.segment_samples = int(segment_duration * self.sample_rate)

        self.samples_root = Path(samples_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        (self.output_dir / 'stft_specs').mkdir(exist_ok=True)
        (self.output_dir / 'mel_specs').mkdir(exist_ok=True)
        (self.output_dir / 'audio_segments').mkdir(exist_ok=True)

        self.global_stats_path = self.output_dir / 'global_normalization_stats.json'
        self.global_stats = None

        with open('selected_texture_samples.json', 'r') as f:
            selection_data = json.load(f)

        self.texture_lookup = {}
        for sample in selection_data['sample_details']:
            self.texture_lookup[sample['filename']] = sample['texture']
        print(f"Loaded original texture values for {len(self.texture_lookup)} files")

    def compute_global_stats_efficient(self, all_file_paths, n_bins=10000):
        print("\n=== Computing Global Normalization Statistics ===")

        min_samples = self.n_fft + self.hop_size

        sample_size = min(5, len(all_file_paths))
        sample_paths = random.sample(all_file_paths, sample_size)

        all_min, all_max = float('inf'), float('-inf')

        for file_info in sample_paths:
            try:
                y, sr = librosa.load(file_info['path'], sr=None, mono=True, dtype=np.float32)
                if sr != self.sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)

                chunk_length = min(len(y), self.sample_rate * 5)
                if chunk_length < min_samples:
                    continue

                chunk = y[:chunk_length]
                stft = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.win_size)
                mag = np.abs(stft).astype(np.float32)

                sqrt_mag = np.sqrt(mag)
                log_mag = np.log1p(sqrt_mag * self.alpha)

                all_min = min(all_min, log_mag.min())
                all_max = max(all_max, log_mag.max())
            except Exception as e:
                print(f"Error sampling {file_info['path']}: {e}")
                continue

        if all_min == float('inf') or all_max == float('-inf'):
            print("Warning: Could not establish valid range from samples, using defaults")
            all_min, all_max = 0.0, 10.0

        hist_range = (all_min - 2.0, all_max + 2.0)
        histogram = np.zeros(n_bins, dtype=np.float32)
        bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1, dtype=np.float32)

        for file_info in tqdm(all_file_paths, desc="Building histogram"):
            try:
                y, sr = librosa.load(file_info['path'], sr=None, mono=True, dtype=np.float32)
                if sr != self.sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)

                chunk_size = self.sample_rate * 30
                for i in range(0, len(y), chunk_size):
                    chunk = y[i:i + chunk_size]

                    if len(chunk) < min_samples:
                        continue

                    stft = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.win_size)
                    mag = np.abs(stft).astype(np.float32)

                    sqrt_mag = np.sqrt(mag)
                    log_mag = np.log1p(sqrt_mag * self.alpha)

                    hist, _ = np.histogram(log_mag.flatten(), bins=bin_edges)
                    histogram += hist.astype(np.float32)

            except Exception as e:
                print(f"Error in global stats for {file_info['path']}: {e}")

        cumsum = np.cumsum(histogram)
        total = cumsum[-1]

        if total == 0:
            print("ERROR: No valid data processed for histogram!")
            return None

        p_low = 0.01
        p_high = 99.99

        low_idx = np.searchsorted(cumsum, total * p_low / 100)
        high_idx = np.searchsorted(cumsum, total * p_high / 100)

        low_idx = np.clip(low_idx, 0, n_bins - 1)
        high_idx = np.clip(high_idx, 0, n_bins - 1)

        global_low = float(bin_edges[low_idx])
        global_high = float(bin_edges[high_idx])

        global_stats = {
            'global_low': global_low,
            'global_high': global_high,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'compression': 'sqrt',
            'percentiles': [p_low, p_high],
            'n_fft': self.n_fft,
            'hop_length': self.hop_size,
            'win_length': self.win_size,
            'sample_rate': self.sample_rate,
            'histogram_range': [float(hist_range[0]), float(hist_range[1])],
            'computed_from_n_files': len(all_file_paths)
        }

        with open(self.global_stats_path, 'w') as f:
            json.dump(global_stats, f, indent=2)

        print(f"\nGlobal stats saved to: {self.global_stats_path}")
        print(f"  Low (0.01%): {global_low:.3f}")
        print(f"  High (99.99%): {global_high:.3f}")
        print(f"  Range: {global_high - global_low:.3f}")

        self.global_stats = global_stats
        return global_stats

    def normalize_stft_with_global_stats(self, stft_complex):
        if self.global_stats is None:
            raise ValueError("Global stats not computed yet!")

        mag = np.abs(stft_complex).astype(np.float32)

        sqrt_mag = np.sqrt(mag)
        log_mag = np.log1p(sqrt_mag * self.global_stats['alpha'])

        norm = np.clip(
            (log_mag - self.global_stats['global_low']) /
            (self.global_stats['global_high'] - self.global_stats['global_low'] + 1e-8),
            0, 1
        ).astype(np.float32)

        gamma_corrected = np.power(norm, self.global_stats['gamma'])

        return gamma_corrected

    def create_fade(self, audio, fade_in=True):
        fade_samples = int(self.fade_duration * self.sample_rate)
        fade_curve = np.linspace(0, 1, fade_samples, dtype=np.float32)

        if fade_in:
            audio[:fade_samples] *= fade_curve
        else:
            audio[-fade_samples:] *= fade_curve[::-1]

        return audio

    def collect_audio_files(self, selected_samples_json='selected_texture_samples.json'):
        with open(selected_samples_json, 'r') as f:
            selection_data = json.load(f)

        sample_details = selection_data['sample_details']
        print(f"Loaded {len(sample_details)} pre-selected samples from {selected_samples_json}")

        selected_filenames = [s['filename'] for s in sample_details]
        all_files = []
        not_found = []

        for filename in selected_filenames:
            found = False
            for audio_file in self.samples_root.rglob(filename):
                all_files.append({
                    'path': audio_file,
                    'filename': audio_file.name
                })
                found = True
                break

            if not found:
                not_found.append(filename)

        if not_found:
            print(f"\nWarning: Could not find {len(not_found)} files:")
            for nf in not_found[:5]:
                print(f"  - {nf}")
            if len(not_found) > 5:
                print(f"  ... and {len(not_found) - 5} more")

        print(f"\nReady to process {len(all_files)} files")
        return all_files

    def process_audio_file(self, file_info):
        try:
            audio_path = file_info['path']

            min_samples = self.n_fft + self.hop_size

            y, orig_sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)

            if orig_sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sample_rate)

            total_samples = len(y)
            if total_samples >= self.segment_samples:
                center_start = (total_samples - self.segment_samples) // 2
                segment = y[center_start:center_start + self.segment_samples].copy()
            else:
                segment = np.pad(y, (0, self.segment_samples - total_samples), mode='constant')

            segment = self.create_fade(segment, fade_in=True)
            segment = self.create_fade(segment, fade_in=False)

            stft_complex = librosa.stft(segment, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.win_size)

            stft_normalized = self.normalize_stft_with_global_stats(stft_complex)

            filename = audio_path.name
            texture_value = self.texture_lookup.get(filename, 0.5)

            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            phase = np.angle(stft_complex)

            return {
                'audio': segment,
                'stft_normalized': stft_normalized,
                'stft_phase': phase,
                'mel_spec': mel_spec_db,
                'texture_value': texture_value,
                'source_file': audio_path.name
            }
        except Exception as e:
            print(f"Error in process_audio_file for {file_info['path']}: {e}")
            return None

    def build_dataset(self):
        selected_files = self.collect_audio_files(
            selected_samples_json='selected_texture_samples.json'
        )

        if not self.global_stats_path.exists():
            stats = self.compute_global_stats_efficient(selected_files)
            if stats is None:
                print("ERROR: Failed to compute global stats!")
                return
        else:
            print(f"Loading existing global stats from {self.global_stats_path}")
            with open(self.global_stats_path, 'r') as f:
                self.global_stats = json.load(f)

        all_segments = []

        for file_info in tqdm(selected_files, desc="Processing audio with global norm"):
            segment_data = self.process_audio_file(file_info)
            if segment_data is not None:
                all_segments.append(segment_data)

        if not all_segments:
            print("ERROR: No segments were successfully processed!")
            return

        texture_values = [s['texture_value'] for s in all_segments]
        print(f"\nTexture value statistics:")
        print(f"  Min: {min(texture_values):.6f}")
        print(f"  Max: {max(texture_values):.6f}")
        print(f"  Mean: {np.mean(texture_values):.6f}")
        print(f"  Std: {np.std(texture_values):.6f}")

        metadata = {
            'audio_params': {
                'sample_rate': self.sample_rate,
                'n_fft': self.n_fft,
                'hop_size': self.hop_size,
                'win_size': self.win_size,
                'stft_shape': [self.n_fft // 2 + 1, None]
            },
            'bigvgan_params': {
                'n_mels': self.n_mels,
                'fmin': self.fmin,
                'fmax': self.fmax or self.sample_rate // 2
            },
            'normalization': 'global',
            'global_stats_file': 'global_normalization_stats.json',
            'processing_params': {
                'segment_duration': self.segment_duration,
                'fade_duration': self.fade_duration,
                'texture_window': 3
            },
            'segments': []
        }

        for i, segment in enumerate(tqdm(all_segments, desc="Saving segments")):
            stft_path = self.output_dir / 'stft_specs' / f'stft_{i:06d}.npz'
            np.savez_compressed(
                stft_path,
                magnitude=segment['stft_normalized'].astype(np.float32),
                phase=segment['stft_phase'].astype(np.float32)
            )

            mel_path = self.output_dir / 'mel_specs' / f'mel_{i:06d}.npy'
            np.save(mel_path, segment['mel_spec'].astype(np.float32))

            audio_path = self.output_dir / 'audio_segments' / f'audio_{i:06d}.wav'
            sf.write(audio_path, segment['audio'], self.sample_rate)

            metadata['segments'].append({
                'id': i,
                'stft_path': str(stft_path.relative_to(self.output_dir)),
                'mel_spec_path': str(mel_path.relative_to(self.output_dir)),
                'audio_path': str(audio_path.relative_to(self.output_dir)),
                'texture_value': float(segment['texture_value']),
                'source_file': segment['source_file']
            })

        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset built successfully!")
        print(f"Total segments: {len(all_segments)}")
        print(f"Successfully processed: {len(all_segments)}/{len(selected_files)}")
        print(f"Output directory: {self.output_dir}")
        print(f"  - High-res STFTs in: stft_specs/ (for CAK training)")
        print(f"  - Mel specs in: mel_specs/ (for BigVGAN)")
        print(f"  - Global normalization ensures consistent scaling!")


if __name__ == "__main__":
    builder = CAKDatasetBuilder(
        samples_root='./samples',
        output_dir='./cak_dataset',
        segment_duration=15.0,
        fade_duration=2.0
    )

    builder.build_dataset()
