"""
Network simulation for all-to-all communication in MoE models.
Simulates communication latency and analyzes impact of expert imbalance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class NetworkConfig:
    """Network configuration parameters."""
    bandwidth: float = 100e9  # 100 Gbps in bps
    latency: float = 1e-6  # 1 microsecond base latency
    num_gpus: int = 8
    packet_size: int = 1500  # bytes
    switch_delay: float = 0.5e-6  # 0.5 microseconds
    
    
@dataclass
class ExpertAssignment:
    """Expert assignment for a batch of tokens."""
    expert_indices: np.ndarray  # [num_tokens, top_k]
    token_sizes: np.ndarray  # [num_tokens] - embedding size per token
    

class AllToAllSimulator:
    """Simulates all-to-all communication for MoE expert assignment."""
    
    def __init__(self, config: NetworkConfig):
        """
        Initialize simulator.
        
        Args:
            config: Network configuration
        """
        self.config = config
        
    def compute_data_volume_per_expert(self, 
                                       assignment: ExpertAssignment,
                                       num_experts: int) -> np.ndarray:
        """
        Compute total data volume sent to each expert.
        
        Args:
            assignment: Expert assignment
            num_experts: Total number of experts
            
        Returns:
            Array of data volumes per expert [num_experts]
        """
        data_per_expert = np.zeros(num_experts)
        
        for token_idx, experts in enumerate(assignment.expert_indices):
            token_size = assignment.token_sizes[token_idx]
            for expert_idx in experts:
                data_per_expert[expert_idx] += token_size
        
        return data_per_expert
    
    def simulate_all_to_all(self,
                           data_per_expert: np.ndarray,
                           source_gpu: int = 0) -> Dict:
        """
        Simulate all-to-all communication from one GPU.
        
        Args:
            data_per_expert: Data volume to send to each expert [num_experts]
            source_gpu: Source GPU index
            
        Returns:
            Dictionary with communication metrics
        """
        num_experts = len(data_per_expert)
        experts_per_gpu = num_experts // self.config.num_gpus
        
        # Compute data volume to each GPU
        data_per_gpu = np.zeros(self.config.num_gpus)
        for expert_idx, data_volume in enumerate(data_per_expert):
            target_gpu = expert_idx // experts_per_gpu
            data_per_gpu[target_gpu] += data_volume
        
        # Compute transfer times
        transfer_times = data_per_gpu * 8 / self.config.bandwidth  # bytes to bits
        
        # Add latency components
        latencies = np.full(self.config.num_gpus, self.config.latency)
        latencies += self.config.switch_delay
        
        # Total communication time per destination
        comm_times = transfer_times + latencies
        
        # Maximum time (bottleneck)
        max_comm_time = np.max(comm_times)
        
        # Load imbalance
        avg_comm_time = np.mean(comm_times)
        load_imbalance = (max_comm_time - avg_comm_time) / (avg_comm_time + 1e-10)
        
        return {
            'max_comm_time': float(max_comm_time),
            'avg_comm_time': float(avg_comm_time),
            'min_comm_time': float(np.min(comm_times)),
            'load_imbalance': float(load_imbalance),
            'data_per_gpu': data_per_gpu.tolist(),
            'comm_times': comm_times.tolist()
        }
    
    def simulate_full_batch(self,
                           assignment: ExpertAssignment,
                           num_experts: int) -> Dict:
        """
        Simulate all-to-all for all GPUs in the batch.
        
        Args:
            assignment: Expert assignment
            num_experts: Total number of experts
            
        Returns:
            Aggregated communication metrics
        """
        data_per_expert = self.compute_data_volume_per_expert(assignment, num_experts)
        
        # Assume data is distributed across GPUs
        tokens_per_gpu = len(assignment.expert_indices) // self.config.num_gpus
        
        all_metrics = []
        for gpu_idx in range(self.config.num_gpus):
            # Get assignment for this GPU's tokens
            start_token = gpu_idx * tokens_per_gpu
            end_token = start_token + tokens_per_gpu if gpu_idx < self.config.num_gpus - 1 else len(assignment.expert_indices)
            
            gpu_assignment = ExpertAssignment(
                expert_indices=assignment.expert_indices[start_token:end_token],
                token_sizes=assignment.token_sizes[start_token:end_token]
            )
            
            gpu_data_per_expert = self.compute_data_volume_per_expert(gpu_assignment, num_experts)
            metrics = self.simulate_all_to_all(gpu_data_per_expert, source_gpu=gpu_idx)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        max_times = [m['max_comm_time'] for m in all_metrics]
        avg_times = [m['avg_comm_time'] for m in all_metrics]
        imbalances = [m['load_imbalance'] for m in all_metrics]
        
        return {
            'overall_max_time': float(np.max(max_times)),
            'overall_avg_time': float(np.mean(avg_times)),
            'overall_load_imbalance': float(np.mean(imbalances)),
            'per_gpu_metrics': all_metrics,
            'total_data_per_expert': data_per_expert.tolist()
        }
    
    def compare_assignments(self,
                           baseline_assignment: ExpertAssignment,
                           modified_assignment: ExpertAssignment,
                           num_experts: int) -> Dict:
        """
        Compare communication performance between two assignments.
        
        Args:
            baseline_assignment: Original expert assignment
            modified_assignment: Modified expert assignment
            num_experts: Total number of experts
            
        Returns:
            Comparison metrics
        """
        baseline_metrics = self.simulate_full_batch(baseline_assignment, num_experts)
        modified_metrics = self.simulate_full_batch(modified_assignment, num_experts)
        
        improvement = {
            'max_time_improvement': (baseline_metrics['overall_max_time'] - 
                                    modified_metrics['overall_max_time']) / baseline_metrics['overall_max_time'] * 100,
            'avg_time_improvement': (baseline_metrics['overall_avg_time'] - 
                                    modified_metrics['overall_avg_time']) / baseline_metrics['overall_avg_time'] * 100,
            'imbalance_improvement': (baseline_metrics['overall_load_imbalance'] - 
                                     modified_metrics['overall_load_imbalance']) / baseline_metrics['overall_load_imbalance'] * 100
        }
        
        return {
            'baseline': baseline_metrics,
            'modified': modified_metrics,
            'improvement': improvement
        }


def generate_synthetic_assignment(num_tokens: int,
                                  num_experts: int,
                                  top_k: int = 2,
                                  embedding_size: int = 4096,
                                  imbalance_factor: float = 0.3) -> ExpertAssignment:
    """
    Generate synthetic expert assignment with controlled imbalance.
    
    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        top_k: Experts per token
        embedding_size: Token embedding size in bytes
        imbalance_factor: 0 = balanced, 1 = highly imbalanced
        
    Returns:
        Synthetic expert assignment
    """
    # Create biased probabilities
    probs = np.random.exponential(scale=1.0, size=num_experts)
    # Add imbalance
    probs = probs ** (1 + imbalance_factor * 2)
    probs = probs / probs.sum()
    
    # Sample expert assignments
    expert_indices = np.zeros((num_tokens, top_k), dtype=int)
    for i in range(num_tokens):
        expert_indices[i] = np.random.choice(num_experts, size=top_k, replace=False, p=probs)
    
    token_sizes = np.full(num_tokens, embedding_size)
    
    return ExpertAssignment(expert_indices=expert_indices, token_sizes=token_sizes)


def plot_communication_comparison(comparison: Dict, 
                                 output_path: str = "results/plots/communication_comparison.png"):
    """
    Plot comparison of communication metrics.
    
    Args:
        comparison: Comparison dictionary from simulator
        output_path: Path to save plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    baseline = comparison['baseline']
    modified = comparison['modified']
    improvement = comparison['improvement']
    
    # Max communication time
    ax = axes[0, 0]
    times = [baseline['overall_max_time'] * 1e6, modified['overall_max_time'] * 1e6]
    ax.bar(['Baseline', 'Modified'], times, color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Max Communication Time (μs)', fontsize=12)
    ax.set_title(f'Maximum Communication Time\nImprovement: {improvement["max_time_improvement"]:.2f}%', 
                fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Average communication time
    ax = axes[0, 1]
    times = [baseline['overall_avg_time'] * 1e6, modified['overall_avg_time'] * 1e6]
    ax.bar(['Baseline', 'Modified'], times, color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Avg Communication Time (μs)', fontsize=12)
    ax.set_title(f'Average Communication Time\nImprovement: {improvement["avg_time_improvement"]:.2f}%', 
                fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Load imbalance
    ax = axes[1, 0]
    imbalances = [baseline['overall_load_imbalance'], modified['overall_load_imbalance']]
    ax.bar(['Baseline', 'Modified'], imbalances, color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Load Imbalance', fontsize=12)
    ax.set_title(f'Load Imbalance\nImprovement: {improvement["imbalance_improvement"]:.2f}%', 
                fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Data per expert distribution
    ax = axes[1, 1]
    baseline_data = np.array(baseline['total_data_per_expert']) / 1e6  # MB
    modified_data = np.array(modified['total_data_per_expert']) / 1e6  # MB
    
    x = np.arange(len(baseline_data))
    width = 0.35
    ax.bar(x - width/2, baseline_data, width, label='Baseline', alpha=0.7, color='red')
    ax.bar(x + width/2, modified_data, width, label='Modified', alpha=0.7, color='green')
    ax.set_xlabel('Expert Index', fontsize=12)
    ax.set_ylabel('Data Volume (MB)', fontsize=12)
    ax.set_title('Data Distribution Per Expert', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Communication comparison plot saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    config = NetworkConfig(
        bandwidth=100e9,  # 100 Gbps
        latency=1e-6,  # 1 μs
        num_gpus=8
    )
    
    simulator = AllToAllSimulator(config)
    
    # Generate assignments
    num_tokens = 1024
    num_experts = 16
    top_k = 2
    
    print("Generating synthetic assignments...")
    baseline = generate_synthetic_assignment(num_tokens, num_experts, top_k, imbalance_factor=0.5)
    balanced = generate_synthetic_assignment(num_tokens, num_experts, top_k, imbalance_factor=0.0)
    
    print("Simulating communication...")
    comparison = simulator.compare_assignments(baseline, balanced, num_experts)
    
    print("\nSimulation Results:")
    print(f"Baseline max time: {comparison['baseline']['overall_max_time']*1e6:.2f} μs")
    print(f"Modified max time: {comparison['modified']['overall_max_time']*1e6:.2f} μs")
    print(f"Improvement: {comparison['improvement']['max_time_improvement']:.2f}%")
    
    # Save results
    with open("results/simulation_results.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    plot_communication_comparison(comparison)
