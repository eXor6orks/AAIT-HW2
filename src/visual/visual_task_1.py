import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

EXP_DIR = Path("experiments/remixmatch_optimized")
VIZ_DIR = EXP_DIR / "visualizations"
EXP_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
NUM_CLASSES = 100

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =========================
# VISUALIZATIONS
# =========================
def plot_training_curves(history, epoch):
    """Courbes d'entraînement complètes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = np.arange(1, len(history['loss_total']) + 1)
    
    # Loss totale
    axes[0, 0].plot(epochs, history['loss_total'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Losses supervisée vs non-supervisée
    axes[0, 1].plot(epochs, history['loss_supervised'], 'g-', linewidth=2, label='Supervised')
    axes[0, 1].plot(epochs, history['loss_unsupervised'], 'r-', linewidth=2, label='Unsupervised')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Supervised vs Unsupervised Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Mask rate
    axes[1, 0].plot(epochs, history['mask_rate'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mask Rate')
    axes[1, 0].set_title('Pseudo-label Acceptance Rate', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Target: 0.3+')
    axes[1, 0].legend()
    
    # Learning rate + Threshold
    axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=2, label='LR')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate', color='orange')
    axes[1, 1].tick_params(axis='y', labelcolor='orange')
    axes[1, 1].grid(True, alpha=0.3)
    
    ax2 = axes[1, 1].twinx()
    ax2.plot(epochs, history['threshold'], 'cyan', linewidth=2, label='Threshold')
    ax2.set_ylabel('Pseudo Threshold', color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    
    axes[1, 1].set_title('Learning Rate & Threshold Schedule', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'training_curves_epoch_{epoch}.png', dpi=150)
    plt.close()

def plot_pseudo_distribution(pseudo_counts, epoch):
    """Distribution des pseudo-labels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    classes = np.arange(NUM_CLASSES)
    
    # Bar chart
    bars = ax1.bar(classes, pseudo_counts, color='steelblue', alpha=0.7, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars, pseudo_counts)):
        if count == 0:
            bar.set_color('red')
            bar.set_alpha(0.3)
    
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Pseudo-label Distribution (Epoch {epoch})', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    zero_count = np.sum(pseudo_counts == 0)
    stats_text = f'Total: {pseudo_counts.sum():.0f}\nMean: {pseudo_counts.mean():.1f}\nZero classes: {zero_count}/{NUM_CLASSES}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Heatmap
    n_cols = 10
    n_rows = NUM_CLASSES // n_cols
    counts_grid = pseudo_counts.reshape(n_rows, n_cols)
    
    im = ax2.imshow(counts_grid, cmap='YlOrRd', aspect='auto')
    ax2.set_title('Heatmap View', fontweight='bold')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'pseudo_distribution_epoch_{epoch}.png', dpi=150)
    plt.close()

def plot_distribution_alignment(target_distributions, epoch):
    """Évolution du distribution alignment"""
    target_array = np.array(target_distributions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap évolution
    im = ax1.imshow(target_array.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Class')
    ax1.set_title('Distribution Alignment Evolution', fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Probability')
    
    # Distribution finale
    final_dist = target_array[-1]
    classes = np.arange(NUM_CLASSES)
    ax2.bar(classes, final_dist, color='purple', alpha=0.7)
    ax2.axhline(1.0 / NUM_CLASSES, color='red', linestyle='--', label='Uniform')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('p_model')
    ax2.set_title(f'Final Distribution (Epoch {epoch})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'distribution_alignment_epoch_{epoch}.png', dpi=150)
    plt.close()