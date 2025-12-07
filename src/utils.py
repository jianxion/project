"""
Utility functions for the MoE project.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str):
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate units."""
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics."""
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99))
    }


def plot_loss_curves(history_file: str, output_path: str = "results/plots/loss_curves.png"):
    """Plot training loss curves from history file."""
    history = load_json(history_file)
    
    steps = [entry['step'] for entry in history]
    total_loss = [entry['total_loss'] for entry in history]
    lm_loss = [entry['lm_loss'] for entry in history]
    imbalance_loss = [entry['imbalance_loss'] for entry in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Total and LM loss
    ax1.plot(steps, total_loss, label='Total Loss', linewidth=2)
    ax1.plot(steps, lm_loss, label='LM Loss', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Imbalance loss
    ax2.plot(steps, imbalance_loss, label='Imbalance Loss', color='red', linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Imbalance Loss')
    ax2.set_title('Expert Imbalance Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Loss curves saved to {output_path}")


def create_results_summary(results_dir: str = "results"):
    """Create a comprehensive summary of all results."""
    results_dir = Path(results_dir)
    
    summary = {
        'profiling': {},
        'assignment_tests': {},
        'finetuning': {},
        'simulation': {}
    }
    
    # Load profiling results
    profiling_stats = results_dir / "profiling" / "expert_assignment_stats.json"
    if profiling_stats.exists():
        data = load_json(str(profiling_stats))
        gini_coeffs = [layer['gini_coefficient'] for layer in data.values()]
        summary['profiling'] = {
            'avg_gini': np.mean(gini_coeffs),
            'max_gini': np.max(gini_coeffs),
            'num_layers': len(data)
        }
    
    # Load assignment test results
    assignment_results = results_dir / "assignment_tests" / "assignment_test_results.json"
    if assignment_results.exists():
        data = load_json(str(assignment_results))
        summary['assignment_tests'] = {
            'baseline_perplexity': data.get('baseline', {}).get('wikitext', {}).get('perplexity'),
            'strategies_tested': [k for k in data.keys() if k != 'baseline']
        }
    
    # Load simulation results
    sim_results = results_dir / "simulation" / "simulation_results.json"
    if sim_results.exists():
        data = load_json(str(sim_results))
        summary['simulation'] = {
            'imbalance_levels': list(data.keys()),
            'performance_impact': {}
        }
        for key, metrics in data.items():
            summary['simulation']['performance_impact'][key] = {
                'max_time_us': metrics['overall_max_time'] * 1e6,
                'load_imbalance': metrics['overall_load_imbalance']
            }
    
    # Save summary
    save_json(summary, str(results_dir / "overall_summary.json"))
    
    return summary


if __name__ == "__main__":
    print("Utility module - import and use in other scripts")
