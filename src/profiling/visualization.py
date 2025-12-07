"""
Visualization utilities for expert assignment analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List


def plot_expert_distribution(stats: Dict, output_dir: str = "results/plots"):
    """
    Plot expert assignment distributions for each layer.
    
    Args:
        stats: Statistics dictionary from profiler
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_name, layer_stats in stats.items():
        expert_probs = layer_stats['expert_probs']
        num_experts = len(expert_probs)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar plot of expert probabilities
        x = np.arange(num_experts)
        bars = ax.bar(x, expert_probs, alpha=0.7, edgecolor='black')
        
        # Add uniform distribution line
        uniform_prob = 1.0 / num_experts
        ax.axhline(y=uniform_prob, color='r', linestyle='--', label='Uniform distribution')
        
        # Color bars by deviation from uniform
        for i, bar in enumerate(bars):
            deviation = abs(expert_probs[i] - uniform_prob) / uniform_prob
            if deviation > 0.5:
                bar.set_color('red')
            elif deviation > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        ax.set_xlabel('Expert Index', fontsize=12)
        ax.set_ylabel('Assignment Probability', fontsize=12)
        ax.set_title(f'Expert Assignment Distribution: {layer_name}\n'
                    f'Gini: {layer_stats["gini_coefficient"]:.3f}, '
                    f'CV: {layer_stats["coefficient_of_variation"]:.3f}', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Save plot
        safe_name = layer_name.replace('/', '_').replace('.', '_')
        plt.tight_layout()
        plt.savefig(output_dir / f'expert_dist_{safe_name}.png', dpi=300)
        plt.close()
    
    print(f"Expert distribution plots saved to {output_dir}")


def plot_imbalance_metrics(stats: Dict, output_path: str = "results/plots/imbalance_metrics.png"):
    """
    Plot imbalance metrics across all layers.
    
    Args:
        stats: Statistics dictionary from profiler
        output_path: Path to save the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    layer_names = list(stats.keys())
    gini_coeffs = [stats[name]['gini_coefficient'] for name in layer_names]
    cvs = [stats[name]['coefficient_of_variation'] for name in layer_names]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gini coefficients
    x = np.arange(len(layer_names))
    ax1.bar(x, gini_coeffs, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_ylabel('Gini Coefficient', fontsize=12)
    ax1.set_title('Expert Imbalance Across Layers (Gini Coefficient)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    ax1.axhline(y=0.1, color='orange', linestyle='--', label='Threshold (0.1)')
    ax1.axhline(y=0.2, color='red', linestyle='--', label='High imbalance (0.2)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Coefficient of variation
    ax2.bar(x, cvs, alpha=0.7, edgecolor='black', color='coral')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Coefficient of Variation', fontsize=12)
    ax2.set_title('Expert Imbalance Across Layers (CV)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Imbalance metrics plot saved to {output_path}")


def plot_heatmap_across_layers(stats: Dict, output_path: str = "results/plots/expert_heatmap.png"):
    """
    Create a heatmap showing expert usage across layers.
    
    Args:
        stats: Statistics dictionary from profiler
        output_path: Path to save the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    layer_names = list(stats.keys())
    max_experts = max(len(stats[name]['expert_probs']) for name in layer_names)
    
    # Create matrix: rows = layers, columns = experts
    matrix = np.zeros((len(layer_names), max_experts))
    
    for i, name in enumerate(layer_names):
        probs = stats[name]['expert_probs']
        matrix[i, :len(probs)] = probs
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(matrix, 
                xticklabels=range(max_experts),
                yticklabels=[name.split('.')[-1] for name in layer_names],
                cmap='YlOrRd',
                cbar_kws={'label': 'Assignment Probability'},
                annot=False)
    
    plt.xlabel('Expert Index', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title('Expert Assignment Heatmap Across Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Expert heatmap saved to {output_path}")


def create_summary_report(stats: Dict, summary: Dict, output_path: str = "results/profiling_report.txt"):
    """
    Create a text summary report of the profiling results.
    
    Args:
        stats: Statistics dictionary from profiler
        summary: Summary dictionary
        output_path: Path to save the report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Expert Assignment Profiling Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Summary:\n")
        f.write(f"  Number of MoE layers: {summary['num_layers']}\n")
        f.write(f"  Average Gini coefficient: {summary['average_gini']:.4f}\n")
        f.write(f"  Maximum Gini coefficient: {summary['max_gini']:.4f}\n")
        f.write(f"  Average CV: {summary['average_cv']:.4f}\n")
        f.write(f"  Maximum CV: {summary['max_cv']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Per-Layer Statistics:\n")
        f.write("=" * 80 + "\n\n")
        
        for layer_name, layer_stats in stats.items():
            f.write(f"\nLayer: {layer_name}\n")
            f.write(f"  Number of experts: {layer_stats['num_experts']}\n")
            f.write(f"  Total assignments: {layer_stats['total_assignments']}\n")
            f.write(f"  Gini coefficient: {layer_stats['gini_coefficient']:.4f}\n")
            f.write(f"  Std deviation: {layer_stats['std_deviation']:.4f}\n")
            f.write(f"  Max/Min ratio: {layer_stats['max_min_ratio']:.2f}\n")
            f.write(f"  Coefficient of variation: {layer_stats['coefficient_of_variation']:.4f}\n")
            
            # Find most and least used experts
            expert_probs = np.array(layer_stats['expert_probs'])
            most_used = np.argmax(expert_probs)
            least_used = np.argmin(expert_probs)
            
            f.write(f"  Most used expert: #{most_used} ({expert_probs[most_used]:.4f})\n")
            f.write(f"  Least used expert: #{least_used} ({expert_probs[least_used]:.4f})\n")
            f.write("-" * 80 + "\n")
    
    print(f"Summary report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    with open("results/expert_assignment_stats.json", 'r') as f:
        stats = json.load(f)
    
    plot_expert_distribution(stats)
    plot_imbalance_metrics(stats)
    plot_heatmap_across_layers(stats)
