import json
import random
from pathlib import Path


def generate_texture_samples(samples_root='./samples',
                             output_file='selected_texture_samples.json',
                             min_texture=0.1,
                             max_texture=1.0,
                             seed=42):
    """
    Scan audio files and assign random texture values
    Texture values are NEVER 0 to ensure proper training dynamics
    """

    random.seed(seed)

    samples_path = Path(samples_root)
    audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.aif', '.ogg'}

    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(samples_path.rglob(f'*{ext}'))

    print(f"Found {len(audio_files)} audio files in {samples_root}")

    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {samples_root}")
        return

    sample_details = []

    for audio_file in sorted(audio_files):
        texture_value = random.uniform(min_texture, max_texture)

        sample_details.append({
            'filename': audio_file.name,
            'texture': round(texture_value, 6)
        })

    texture_values = [s['texture'] for s in sample_details]

    output_data = {
        'sample_details': sample_details,
        'metadata': {
            'total_samples': len(sample_details),
            'min_texture': min(texture_values),
            'max_texture': max(texture_values),
            'mean_texture': sum(texture_values) / len(texture_values),
            'seed': seed
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Created {output_file}")
    print(f"  Total samples: {len(sample_details)}")
    print(f"  Texture range: [{min(texture_values):.3f}, {max(texture_values):.3f}]")
    print(f"  Mean texture: {sum(texture_values) / len(texture_values):.3f}")

    print(f"\nFirst 5 samples:")
    for sample in sample_details[:5]:
        print(f"  {sample['filename']}: texture={sample['texture']:.3f}")


if __name__ == "__main__":
    generate_texture_samples(
        samples_root='./samples',
        output_file='selected_texture_samples.json',
        min_texture=0.1,
        max_texture=1.0,
        seed=42
    )
