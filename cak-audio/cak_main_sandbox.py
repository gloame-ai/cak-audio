"""
CAK (Conditioning Aware Kernels) Training Script
Research code for "CAK: Emergent Audio Effects from Minimal Deep Learning"

This code implements the training procedure described in our paper using
WGAN-GP and the AuGAN framework. Note: 'grain_density' refers to the
texture/control parameter from earlier experiments.

Author: Austin Rockman (austin@gloame.ai)
Date: July 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ============= HYPERPARAMETERS ===============
BATCH_SIZE = 16
N_CRITIC = 5  # train critic 5x per generator update
LAMBDA_GP = 10.0  # grad penalty weight
LAMBDA_COMP = 2.0  # compliance/violation weight
RECON_WEIGHT = 5.0  # reconstruction loss weight

# temperature annealing for soft gate
TEMP_INITIAL = 2.0
TEMP_FINAL = 20.0
ANNEAL_EPOCHS = 5  # epochs to reach final temperature


# ============= SHARED DETECTOR =============
class SharedTextureDetector(nn.Module):
    """texture detector - shared between G and D for the audit game"""

    def __init__(self):
        super(SharedTextureDetector, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        # small initialization values (gain=0.1) allow the detector (D) to make subtle observations first, 
        # then gradually learn what matters. zero bias ensures the detector starts neutral, 
        # then learn the minimal intervention needed.
        nn.init.xavier_normal_(self.conv.weight, gain=0.1)
        nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x):
        return self.conv(x)


# ============= SOFT-GATE CAK LAYER =============
class SoftGateCAKLayer(nn.Module):
    """
    CAK with temperature-annealed soft gate
    output = x + (detected_texture × texture_value × scale × gate)
    where gate = σ((texture_value - τ) × T)
    """

    def __init__(self, shared_detector):
        super(SoftGateCAKLayer, self).__init__()
        self.detector = shared_detector
        # scale parameter (11th and final learnable parameter)
        # - starts at 1.0: no amplification or reduction initially
        # - network learns optimal scaling during training
        # - global multiplier: affects all patterns equally
        self.scale = nn.Parameter(torch.ones(1) * 1.0) # could simplify to torch.tensor(1.0)

        self.register_buffer('epoch', torch.tensor(0))
        self.register_buffer('threshold', torch.tensor(0.3))  # τ

    @property
    def temperature(self):
        """temperature schedule: TEMP_INITIAL → TEMP_FINAL over ANNEAL_EPOCHS"""
        e = self.epoch.item()
        progress = min(1.0, e / ANNEAL_EPOCHS)
        return TEMP_INITIAL + (TEMP_FINAL - TEMP_INITIAL) * progress

    def soft_gate(self, texture_value):
        """soft gate that helps with identity when texture_value ≈ 0"""
        return torch.sigmoid((texture_value - self.threshold) * self.temperature)

    def forward(self, x, texture_value):
        # detect texture patterns
        texture_patterns = self.detector(x)

        # prepare texture value
        texture_v = texture_value.view(-1, 1, 1, 1)

        # apply soft gate
        gate = self.soft_gate(texture_v)

        # CAK formula with gate (core method outlined in the paper)
        scaled_texture = texture_patterns * texture_v * self.scale * gate
        output = x + scaled_texture

        return output, texture_patterns


# ============= GENERATOR =============
class TextureGenerator(nn.Module):
    """generator with soft-gate CAK layer"""

    def __init__(self, shared_detector):
        super(TextureGenerator, self).__init__()
        self.cak = SoftGateCAKLayer(shared_detector)

    def update_epoch(self, epoch):
        self.cak.epoch.fill_(epoch)

    def forward(self, x, texture_value):
        return self.cak(x, texture_value)


# ============= CRITIC (WGAN-GP STYLE) =============
class TextureCritic(nn.Module):
    """WGAN-GP Critic (D) with audit game"""

    def __init__(self, shared_detector):
        super(TextureCritic, self).__init__()
        self.detector = shared_detector

        # realism branch, outputs raw score - we are aware that WGAN typically does not use Norm layers
        # we have only 200 samples, instance is used to make sure each spec gets normalized to its own statistics
        # just a config choice to prevent potentially weird batch interactions
        self.realism_net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def compute_texture_from_spec(self, x):
        """compute texture level using shared detector"""
        patterns = self.detector(x)
        texture_level = patterns.mean(dim=[2, 3])
        return texture_level.squeeze(1)

    def forward(self, x, claimed_texture):
        # get the raw critic score
        score = self.realism_net(x).squeeze(1) # how real does this look?

        # audit: measure texture
        measured_texture = self.compute_texture_from_spec(x) # what do I measure?

        # violation (also outlined in paper)
        if claimed_texture.dim() > 1:
            claimed_texture = claimed_texture.squeeze() # scalar for comparison
        violation = torch.abs(measured_texture - claimed_texture) # did G lie?

        return score, violation, measured_texture # realness, honesty, actual measurement


# ============= DATASET =============
class TextureDataset(Dataset):
    """dataset loading from preprocessed spectrograms"""

    # all control value scores live in the json, it's ideal to make your own, we do not share the json, but examples
    # of our approach to formatting can be found in dataset_format_example.py
    def __init__(self, metadata_path='cak_bigvgan_dataset/dataset_metadata.json'):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.segments = self.metadata['segments']
        self.data_dir = os.path.dirname(metadata_path)

        print(f"Loaded {len(self.segments)} segments")
        print(f"Min absolute texture: {min(s['grain_density'] for s in self.segments):.3f}")
        print(f"Max absolute texture: {max(s['grain_density'] for s in self.segments):.3f}")

    def __len__(self):
        return len(self.segments) * 2  # each segment used twice for identity pair and transformation pair

    def __getitem__(self, idx):
        if idx % 2 == 0:
            # IDENTITY: input = output, texture = 0
            seg_idx = idx // 2
            segment = self.segments[seg_idx]

            # load STFT
            stft_path = os.path.join(self.data_dir, segment['stft_path'])
            stft_data = np.load(stft_path)
            spec = torch.FloatTensor(stft_data['magnitude']).unsqueeze(0)

            return spec, spec, torch.tensor([0.0], dtype=torch.float32)

        else:
            # TRANSFORMATION: low → high texture
            n_segs = len(self.segments)

            # sort samples by texture to get proper pairs
            sorted_indices = sorted(range(n_segs),
                                    key=lambda i: self.segments[i]['grain_density'])

            # get low and high texture segments
            low_idx = sorted_indices[idx % (n_segs // 2)]
            high_idx = sorted_indices[n_segs // 2 + (idx % (n_segs // 2))]

            low_seg = self.segments[low_idx]
            high_seg = self.segments[high_idx]

            # load STFTs
            low_stft = np.load(os.path.join(self.data_dir, low_seg['stft_path']))
            high_stft = np.load(os.path.join(self.data_dir, high_seg['stft_path']))

            low_spec = torch.FloatTensor(low_stft['magnitude']).unsqueeze(0)
            high_spec = torch.FloatTensor(high_stft['magnitude']).unsqueeze(0)

            # texture control is the difference
            texture_control = high_seg['grain_density'] - low_seg['grain_density']

            return low_spec, high_spec, torch.FloatTensor([texture_control])


# ============= GRADIENT PENALTY =============
def compute_gradient_penalty(critic, real_samples, fake_samples, texture_values):
    """compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)

    # random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    # get critic scores
    scores, _, _ = critic(interpolated, texture_values.detach())  # detach texture_values, important!

    # compute gradients
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]

    # grad penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# ============= TRAINING WITH WGAN-GP =============
def train_wgan_gp(num_epochs=50):
    """train with WGAN-GP loss and soft-gate CAK"""

    # create output directories
    os.makedirs('wgan_texture_output', exist_ok=True)
    os.makedirs('wgan_texture_output/checkpoints', exist_ok=True)
    os.makedirs('wgan_texture_output/samples', exist_ok=True)

    # create SHARED detector
    shared_detector = SharedTextureDetector().to(device)

    # create G and D (critic)
    generator = TextureGenerator(shared_detector).to(device)
    critic = TextureCritic(shared_detector).to(device)

    # dataset
    dataset = TextureDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n=== Dataset Sanity Checks ===")
    all_texture_densities = [seg['grain_density'] for seg in dataset.segments]
    print(f"Texture density range in dataset:")
    print(f"  Min: {min(all_texture_densities):.6f}")
    print(f"  Max: {max(all_texture_densities):.6f}")
    print(f"  Mean: {np.mean(all_texture_densities):.6f}")
    print(f"  Std: {np.std(all_texture_densities):.6f}")

    print("\nFirst batch check:")
    for i, (input_specs, target_specs, texture_values) in enumerate(dataloader):
        print(f"  Batch {i + 1}:")
        print(f"    Input shape: {input_specs.shape}")
        print(f"    Texture values: {texture_values.squeeze().numpy()}")
        print(f"    Texture range: [{texture_values.min().item():.6f}, {texture_values.max().item():.6f}]")
        print(f"    Identity samples (texture≈0): {(texture_values.abs() < 1e-6).sum().item()}/{BATCH_SIZE}")
        break

    print("\nTesting detector response:")
    test_input = torch.randn(1, 1, 1025, 256).to(device)
    test_texture = torch.tensor([0.5]).to(device)
    with torch.no_grad():
        test_output, test_patterns = generator(test_input, test_texture)
        print(f"  Pattern strength: {test_patterns.abs().mean().item():.6f}")
        print(f"  Output change: {(test_output - test_input).abs().mean().item():.6f}")

    print("\n=== Starting WGAN-GP CAK Training ===")
    print(f"Shared detector: {sum(p.numel() for p in shared_detector.parameters())} params")
    print(f"Temperature schedule: {TEMP_INITIAL} → {TEMP_FINAL} over {ANNEAL_EPOCHS} epochs")
    print("=" * 50 + "\n")

    # optimizers - beta1=0.0 is standard WGAN recipe, momentum can interfere with wasserstein grads
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    c_optimizer = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))

    history = {
        'g_loss': [], 'c_loss': [], 'wasserstein_d': [],
        'violations': [], 'pattern_strength': [], 'temperature': [], 'scale': []
    }

    print(f"\n=== WGAN-GP CAK Training with Soft Gate ===")
    print(f"Shared detector: {sum(p.numel() for p in shared_detector.parameters())} params")
    print(f"Temperature schedule: {TEMP_INITIAL} → {TEMP_FINAL} over {ANNEAL_EPOCHS} epochs")

    for epoch in range(num_epochs):
        # update temperature
        generator.update_epoch(epoch)
        current_temp = generator.cak.temperature

        g_losses, c_losses, w_dists, violations, patterns = [], [], [], [], []

        progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} (T={current_temp:.1f})')

        for input_specs, target_specs, texture_values in progress:
            input_specs = input_specs.to(device)
            target_specs = target_specs.to(device)
            texture_values = texture_values.to(device)

            # ========== Train Critic (N_CRITIC times) ==========
            for _ in range(N_CRITIC):
                critic.zero_grad()

                # generate fake
                with torch.no_grad():
                    fake_specs, _ = generator(input_specs, texture_values)

                # critic scores and violations
                real_score, real_viol, _ = critic(target_specs, texture_values)
                fake_score, fake_viol, _ = critic(fake_specs, texture_values)

                # WGAN loss: maximize real - fake
                wasserstein_d = real_score.mean() - fake_score.mean()

                # gradient penalty
                gp = compute_gradient_penalty(critic, target_specs, fake_specs, texture_values)

                # compliance loss on fakes
                compliance_loss = LAMBDA_COMP * fake_viol.mean()

                # total critic loss (outlined in paper)
                c_loss = -wasserstein_d + LAMBDA_GP * gp + compliance_loss

                c_loss.backward()
                c_optimizer.step()

            # ========== Train Generator ==========
            generator.zero_grad()

            # generate
            fake_specs, texture_patterns = generator(input_specs, texture_values)

            # critic evaluation
            fake_score, fake_viol, _ = critic(fake_specs, texture_values)

            # G wants to maximize critic score
            g_adv_loss = -fake_score.mean()

            # G wants to minimize violations
            g_viol_loss = LAMBDA_COMP * fake_viol.mean()

            # reconstruction loss
            g_recon_loss = F.l1_loss(fake_specs, target_specs)

            # pattern regularization (keep detector alive)
            pattern_reg = -0.01 * torch.log(texture_patterns.abs().mean() + 1e-8)

            # total generator loss (outlined in paper)
            g_loss = g_adv_loss + g_viol_loss + RECON_WEIGHT * g_recon_loss + pattern_reg

            g_loss.backward()
            g_optimizer.step()





            """ REMAINING CODE IS JUST VISUALIZATION FUNCTIONS """





            # record metrics
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())
            w_dists.append(wasserstein_d.item())
            violations.append(fake_viol.mean().item())
            patterns.append(texture_patterns.abs().mean().item())

            # progress
            progress.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'C': f'{c_loss.item():.3f}',
                'W': f'{wasserstein_d.item():.3f}',
                'V': f'{fake_viol.mean().item():.3f}'
            })

        # epoch stats
        history['g_loss'].append(np.mean(g_losses))
        history['c_loss'].append(np.mean(c_losses))
        history['wasserstein_d'].append(np.mean(w_dists))
        history['violations'].append(np.mean(violations))
        history['pattern_strength'].append(np.mean(patterns))
        history['temperature'].append(current_temp)
        history['scale'].append(generator.cak.scale.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"G: {history['g_loss'][-1]:.4f}, "
              f"C: {history['c_loss'][-1]:.4f}, "
              f"W-dist: {history['wasserstein_d'][-1]:.4f}, "
              f"Viol: {history['violations'][-1]:.4f}, "
              f"T: {current_temp:.1f}, "
              f"Scale: {generator.cak.scale.item():.3f}")

        if (epoch + 1) % 10 == 0:
            print(f"\n--- Texture Control Check ---")
            random_idx = np.random.randint(0, len(dataset))
            test_input, _, _ = dataset[random_idx]
            test_spec = test_input.unsqueeze(0).to(device)

            generator.eval()
            critic.eval()

            with torch.no_grad():
                test_textures = [-1.0, -0.5, 0.0, 0.5, 1.0]
                for tt in test_textures:
                    texture_tensor = torch.tensor([tt]).to(device)
                    out, pat = generator(test_spec, texture_tensor)
                    change = (out - test_spec).abs().mean().item()
                    gate_value = generator.cak.soft_gate(texture_tensor).item()

                    _, viol, measured = critic(out, texture_tensor)

                    print(f"  Texture {tt:+.1f}: change={change:.6f}, gate={gate_value:.3f}, "
                          f"measured={measured.item():.3f}, viol={viol.item():.3f}")

            print(f"  (Tested on sample {random_idx}/{len(dataset)})")
            print("-" * 40)

        generator.train()
        critic.train()

        # visualize
        if (epoch + 1) % 10 == 0:
            visualize_results(generator, critic, dataloader, epoch + 1)

        # save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'shared_detector': shared_detector.state_dict(),
                'history': history,
                'epoch': epoch
            }, f'wgan_texture_output/checkpoints/epoch_{epoch + 1}.pt')

    # save final
    torch.save({
        'generator': generator.state_dict(),
        'critic': critic.state_dict(),
        'shared_detector': shared_detector.state_dict(),
        'history': history
    }, 'wgan_texture_output/final_wgan_texture.pt')

    plot_wgan_history(history)
    return history


def visualize_results(generator, critic, dataloader, epoch):
    """Visualize WGAN-GP results"""
    generator.eval()
    critic.eval()

    # get a batch
    for input_specs, target_specs, texture_values in dataloader:
        input_specs = input_specs[:5].to(device)
        target_specs = target_specs[:5].to(device)
        texture_values = texture_values[:5].to(device)
        break

    with torch.no_grad():
        # generate
        processed_specs, patterns = generator(input_specs, texture_values)

        # audit
        scores, violations, measured = critic(processed_specs, texture_values)

        # visualize
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        for i in range(5):
            # input
            axes[0, i].imshow(input_specs[i, 0].cpu().numpy(),
                              aspect='auto', origin='lower', cmap='viridis')
            axes[0, i].set_title(f'Input (texture={texture_values[i].item():.2f})')

            # output
            axes[1, i].imshow(processed_specs[i, 0].cpu().numpy(),
                              aspect='auto', origin='lower', cmap='viridis')
            axes[1, i].set_title(f'Output (score={scores[i].item():.2f})')

            # patterns
            pattern = patterns[i, 0].cpu().numpy()
            vmax = np.abs(pattern).max()
            if vmax > 0:
                axes[2, i].imshow(pattern, aspect='auto', origin='lower',
                                  cmap='RdBu', vmin=-vmax, vmax=vmax)
            axes[2, i].set_title(f'Pattern (viol={violations[i].item():.3f})')

        plt.suptitle(f'WGAN-GP Results - Epoch {epoch} (T={generator.cak.temperature:.1f})',
                     fontsize=16)
        plt.tight_layout()
        plt.savefig(f'wgan_texture_output/samples/epoch_{epoch}_wgan.png', dpi=150)
        plt.close()

    generator.train()
    critic.train()


def plot_wgan_history(history):
    """Plot WGAN-GP training history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # losses
    axes[0, 0].plot(history['g_loss'], label='Generator')
    axes[0, 0].plot(history['c_loss'], label='Critic')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # wasserstein distance
    axes[0, 1].plot(history['wasserstein_d'])
    axes[0, 1].set_title('Wasserstein Distance')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # violations
    axes[0, 2].plot(history['violations'], color='red')
    axes[0, 2].set_title('Mean Violations')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True)

    # pattern strength
    axes[1, 0].plot(history['pattern_strength'], color='green')
    axes[1, 0].set_title('Pattern Detection Strength')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True)

    # temperature schedule
    axes[1, 1].plot(history['temperature'], color='orange')
    axes[1, 1].set_title('Soft Gate Temperature')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Temperature')
    axes[1, 1].grid(True)

    # violations vs Temperature
    ax2 = axes[1, 2].twinx()
    axes[1, 2].plot(history['violations'], 'r-', label='Violations')
    ax2.plot(history['temperature'], 'orange', alpha=0.6, label='Temperature')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Violations', color='r')
    ax2.set_ylabel('Temperature', color='orange')
    axes[1, 2].set_title('Violations vs Temperature')
    axes[1, 2].grid(True)

    plt.suptitle('WGAN-GP CAK Training History', fontsize=16)
    plt.tight_layout()
    plt.savefig('wgan_texture_output/training_history.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    # train with WGAN-GP and soft gates
    history = train_wgan_gp(num_epochs=100)
