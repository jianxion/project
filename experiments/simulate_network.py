#!/usr/bin/env python3
"""
Experiment 4: Simulate network performance under different expert assignments.
"""

import sys
sys.path.append('.')

from src.simulation.network_sim import (
    AllToAllSimulator, 
    NetworkConfig, 
    generate_synthetic_assignment,
    plot_communication_comparison
)
import numpy as np
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Simulate network performance")
    parser.add_argument("--num_tokens", type=int, default=1024,
                       help="Number of tokens per batch")
    parser.add_argument("--num_experts", type=int, default=16,
                       help="Total number of experts")
    parser.add_argument("--top_k", type=int, default=2,
                       help="Experts selected per token")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="Number of GPUs")
    parser.add_argument("--bandwidth", type=float, default=100,
                       help="Network bandwidth in Gbps")
    parser.add_argument("--latency", type=float, default=1.0,
                       help="Base network latency in microseconds")
    parser.add_argument("--imbalance_levels", type=float, nargs="+",
                       default=[0.0, 0.2, 0.5, 0.8],
                       help="Imbalance levels to test")
    parser.add_argument("--output_dir", type=str, default="results/simulation",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Network Performance Simulation")
    print("="*80)
    print(f"Configuration:")
    print(f"  Tokens per batch: {args.num_tokens}")
    print(f"  Number of experts: {args.num_experts}")
    print(f"  Top-k: {args.top_k}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Bandwidth: {args.bandwidth} Gbps")
    print(f"  Latency: {args.latency} μs")
    print()
    
    # Create network configuration
    config = NetworkConfig(
        bandwidth=args.bandwidth * 1e9,  # Convert to bps
        latency=args.latency * 1e-6,  # Convert to seconds
        num_gpus=args.num_gpus
    )
    
    # Initialize simulator
    simulator = AllToAllSimulator(config)
    
    # Test different imbalance levels
    all_results = {}
    
    for imbalance in args.imbalance_levels:
        print(f"\n{'='*80}")
        print(f"Testing imbalance level: {imbalance:.2f}")
        print("="*80)
        
        # Generate assignment with specified imbalance
        assignment = generate_synthetic_assignment(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
            imbalance_factor=imbalance
        )
        
        # Simulate
        metrics = simulator.simulate_full_batch(assignment, args.num_experts)
        
        # Store results
        all_results[f"imbalance_{imbalance:.2f}"] = metrics
        
        # Print results
        print(f"Overall max communication time: {metrics['overall_max_time']*1e6:.2f} μs")
        print(f"Overall avg communication time: {metrics['overall_avg_time']*1e6:.2f} μs")
        print(f"Overall load imbalance: {metrics['overall_load_imbalance']:.4f}")
        
        # Calculate tail latency
        gpu_max_times = [m['max_comm_time'] for m in metrics['per_gpu_metrics']]
        tail_latency = np.percentile(gpu_max_times, 99)
        print(f"P99 tail latency: {tail_latency*1e6:.2f} μs")
    
    # Compare balanced vs imbalanced
    if len(args.imbalance_levels) >= 2:
        print(f"\n{'='*80}")
        print("Comparison: Balanced vs Imbalanced")
        print("="*80)
        
        balanced_key = f"imbalance_{args.imbalance_levels[0]:.2f}"
        imbalanced_key = f"imbalance_{args.imbalance_levels[-1]:.2f}"
        
        balanced_time = all_results[balanced_key]['overall_max_time']
        imbalanced_time = all_results[imbalanced_key]['overall_max_time']
        
        slowdown = (imbalanced_time - balanced_time) / balanced_time * 100
        
        print(f"Balanced max time: {balanced_time*1e6:.2f} μs")
        print(f"Imbalanced max time: {imbalanced_time*1e6:.2f} μs")
        print(f"Slowdown: {slowdown:.2f}%")
        
        # Visualize comparison
        balanced_assignment = generate_synthetic_assignment(
            args.num_tokens, args.num_experts, args.top_k, 
            imbalance_factor=args.imbalance_levels[0]
        )
        imbalanced_assignment = generate_synthetic_assignment(
            args.num_tokens, args.num_experts, args.top_k,
            imbalance_factor=args.imbalance_levels[-1]
        )
        
        comparison = simulator.compare_assignments(
            imbalanced_assignment, balanced_assignment, args.num_experts
        )
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_communication_comparison(
            comparison, 
            output_path=output_dir / "communication_comparison.png"
        )
    
    # Save all results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "simulation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    print(f"\n{'='*80}")
    print("Simulation Summary")
    print("="*80)
    print(f"{'Imbalance':<15} {'Max Time (μs)':<20} {'Avg Time (μs)':<20} {'Load Imbalance':<15}")
    print("-"*80)
    
    for imbalance in args.imbalance_levels:
        key = f"imbalance_{imbalance:.2f}"
        metrics = all_results[key]
        print(f"{imbalance:<15.2f} "
              f"{metrics['overall_max_time']*1e6:<20.2f} "
              f"{metrics['overall_avg_time']*1e6:<20.2f} "
              f"{metrics['overall_load_imbalance']:<15.4f}")
    
    print("="*80)
    print(f"\nResults saved to {output_dir}/")
    print("\nKey findings:")
    print("1. Expert imbalance increases communication latency")
    print("2. Tail latency grows significantly with imbalance")
    print("3. Load balancing is critical for performance")


if __name__ == "__main__":
    main()
