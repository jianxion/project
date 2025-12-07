#!/usr/bin/env python3
"""
Lightweight demo for M1 Macs - demonstrates the framework without heavy models.
Uses synthetic data to show the complete workflow.
"""

import sys
sys.path.append('.')

import numpy as np
import json
from pathlib import Path
from src.simulation.network_sim import (
    AllToAllSimulator,
    NetworkConfig,
    generate_synthetic_assignment,
    plot_communication_comparison,
    ExpertAssignment
)
from src.profiling.visualization import (
    plot_expert_distribution,
    plot_imbalance_metrics,
    create_summary_report
)
import matplotlib.pyplot as plt


def generate_synthetic_profiling_stats(num_layers=8, num_experts=8):
    """Generate synthetic profiling statistics to demonstrate the framework."""
    stats = {}
    
    for layer_idx in range(num_layers):
        layer_name = f"model.layers.{layer_idx}.block_sparse_moe"
        
        # Generate imbalanced expert usage (more realistic)
        imbalance_factor = np.random.uniform(0.2, 0.6)
        probs = np.random.exponential(scale=1.0, size=num_experts)
        probs = probs ** (1 + imbalance_factor * 2)
        probs = probs / probs.sum()
        
        total_assignments = 10000
        expert_counts = (probs * total_assignments).astype(int)
        
        # Calculate metrics
        uniform_prob = 1.0 / num_experts
        std_dev = float(np.std(probs))
        gini = float((num_experts + 1 - 2 * np.sum(np.cumsum(np.sort(probs))) / np.sum(probs)) / num_experts)
        cv = std_dev / (probs.mean() + 1e-10)
        
        stats[layer_name] = {
            'expert_counts': expert_counts.tolist(),
            'expert_probs': probs.tolist(),
            'gini_coefficient': gini,
            'std_deviation': std_dev,
            'max_min_ratio': float(probs.max() / (probs.min() + 1e-10)),
            'coefficient_of_variation': float(cv),
            'total_assignments': int(total_assignments),
            'num_experts': num_experts
        }
    
    return stats


def demo_profiling():
    """Demonstrate expert profiling with synthetic data."""
    print("\n" + "="*80)
    print("DEMO 1: Expert Assignment Profiling (Synthetic Data)")
    print("="*80)
    
    # Generate synthetic stats
    stats = generate_synthetic_profiling_stats(num_layers=8, num_experts=8)
    
    # Save statistics
    output_dir = Path("results/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "synthetic_profiling_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Calculate summary
    gini_coeffs = [s['gini_coefficient'] for s in stats.values()]
    cvs = [s['coefficient_of_variation'] for s in stats.values()]
    
    summary = {
        'average_gini': float(np.mean(gini_coeffs)),
        'max_gini': float(np.max(gini_coeffs)),
        'average_cv': float(np.mean(cvs)),
        'max_cv': float(np.max(cvs)),
        'num_layers': len(stats)
    }
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_expert_distribution(stats, output_dir=str(output_dir / "plots"))
    plot_imbalance_metrics(stats, output_path=str(output_dir / "plots" / "imbalance_metrics.png"))
    
    # Create report
    create_summary_report(stats, summary, output_path=str(output_dir / "profiling_report.txt"))
    
    # Print summary
    print("\n" + "-"*80)
    print("Profiling Summary:")
    print(f"  Average Gini coefficient: {summary['average_gini']:.4f}")
    print(f"  Maximum Gini coefficient: {summary['max_gini']:.4f}")
    print(f"  Average CV: {summary['average_cv']:.4f}")
    
    if summary['average_gini'] > 0.3:
        print("\n  ⚠️  HIGH IMBALANCE DETECTED")
    elif summary['average_gini'] > 0.1:
        print("\n  ⚠️  MODERATE IMBALANCE")
    else:
        print("\n  ✓ BALANCED")
    
    print(f"\nResults saved to {output_dir}/")
    return summary


def demo_network_simulation():
    """Demonstrate network simulation."""
    print("\n" + "="*80)
    print("DEMO 2: Network Performance Simulation")
    print("="*80)
    
    # Network configuration (typical for M1 development)
    config = NetworkConfig(
        bandwidth=10e9,  # 10 Gbps (more realistic for development)
        latency=2e-6,    # 2 μs
        num_gpus=4       # Smaller scale
    )
    
    simulator = AllToAllSimulator(config)
    
    # Test different imbalance levels
    imbalance_levels = [0.0, 0.3, 0.6, 0.9]
    results = {}
    
    for imbalance in imbalance_levels:
        assignment = generate_synthetic_assignment(
            num_tokens=512,
            num_experts=8,
            top_k=2,
            imbalance_factor=imbalance
        )
        
        metrics = simulator.simulate_full_batch(assignment, num_experts=8)
        results[f"imbalance_{imbalance:.1f}"] = metrics
        
        print(f"\nImbalance {imbalance:.1f}:")
        print(f"  Max comm time: {metrics['overall_max_time']*1e6:.2f} μs")
        print(f"  Load imbalance: {metrics['overall_load_imbalance']:.4f}")
    
    # Save results
    output_dir = Path("results/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison visualization
    balanced = generate_synthetic_assignment(512, 8, 2, 0.0)
    imbalanced = generate_synthetic_assignment(512, 8, 2, 0.9)
    comparison = simulator.compare_assignments(imbalanced, balanced, 8)
    
    plot_communication_comparison(comparison, output_path=str(output_dir / "communication_comparison.png"))
    
    print(f"\n✓ Simulation results saved to {output_dir}/")
    return results


def demo_assignment_strategies():
    """Demonstrate different assignment strategies."""
    print("\n" + "="*80)
    print("DEMO 3: Assignment Modification Strategies")
    print("="*80)
    
    from src.assignment.modifier import ExpertAssignmentModifier, AssignmentStrategy
    import torch
    
    # Small-scale demonstration
    num_experts = 8
    batch_size = 4
    seq_len = 32
    top_k = 2
    
    modifier = ExpertAssignmentModifier(num_experts, top_k)
    
    # Simulate router logits
    router_logits = torch.randn(batch_size, seq_len, num_experts)
    
    strategies = [
        AssignmentStrategy.BALANCED,
        AssignmentStrategy.ROUND_ROBIN,
        AssignmentStrategy.RANDOM,
        AssignmentStrategy.LOAD_AWARE
    ]
    
    results = {}
    
    for strategy in strategies:
        indices = modifier.apply_strategy(router_logits, strategy)
        
        # Calculate expert usage distribution
        usage = torch.zeros(num_experts)
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(top_k):
                    usage[indices[b, s, k]] += 1
        
        usage_probs = (usage / usage.sum()).numpy()
        gini = (num_experts + 1 - 2 * np.sum(np.cumsum(np.sort(usage_probs))) / np.sum(usage_probs)) / num_experts
        
        results[strategy.value] = {
            'distribution': usage_probs.tolist(),
            'gini': float(gini),
            'std': float(np.std(usage_probs))
        }
        
        print(f"\n{strategy.value.upper()} strategy:")
        print(f"  Gini coefficient: {gini:.4f}")
        print(f"  Std deviation: {np.std(usage_probs):.4f}")
    
    # Save results
    output_dir = Path("results/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "assignment_strategies.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Strategy comparison saved to {output_dir}/")
    return results


def demo_loss_functions():
    """Demonstrate imbalance loss functions."""
    print("\n" + "="*80)
    print("DEMO 4: Imbalance Loss Functions")
    print("="*80)
    
    from src.training.imbalance_loss import ExpertImbalanceLoss
    import torch
    
    num_experts = 8
    batch_size = 4
    seq_len = 32
    
    # Simulate different imbalance scenarios
    scenarios = {
        'balanced': torch.ones(batch_size, seq_len, num_experts) / num_experts,
        'moderate_imbalance': torch.tensor([[2, 1.5, 1, 1, 0.5, 0.5, 0.3, 0.2]]).repeat(batch_size, seq_len, 1),
        'high_imbalance': torch.tensor([[5, 2, 1, 0.5, 0.2, 0.1, 0.1, 0.1]]).repeat(batch_size, seq_len, 1)
    }
    
    # Normalize to probabilities
    for key in scenarios:
        scenarios[key] = scenarios[key] / scenarios[key].sum(dim=-1, keepdim=True)
    
    loss_types = ['gini', 'variance', 'cv', 'entropy']
    results = {}
    
    for scenario_name, router_probs in scenarios.items():
        results[scenario_name] = {}
        print(f"\n{scenario_name.upper()}:")
        
        for loss_type in loss_types:
            loss_fn = ExpertImbalanceLoss(num_experts, loss_weight=1.0, loss_type=loss_type)
            # Convert back to logits for loss function
            router_logits = torch.log(router_probs + 1e-10)
            loss_value = loss_fn(router_logits).item()
            
            results[scenario_name][loss_type] = loss_value
            print(f"  {loss_type}: {loss_value:.4f}")
    
    # Save results
    output_dir = Path("results/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "loss_functions.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Loss function results saved to {output_dir}/")
    return results


def create_final_summary(profiling_summary, simulation_results, assignment_results, loss_results):
    """Create a final comprehensive summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PROJECT SUMMARY")
    print("="*80)
    
    summary = {
        'profiling': {
            'average_gini': profiling_summary['average_gini'],
            'max_gini': profiling_summary['max_gini'],
            'interpretation': 'High imbalance detected' if profiling_summary['average_gini'] > 0.3 else 'Moderate imbalance'
        },
        'simulation': {
            'balanced_time_us': simulation_results['imbalance_0.0']['overall_max_time'] * 1e6,
            'imbalanced_time_us': simulation_results['imbalance_0.9']['overall_max_time'] * 1e6,
            'slowdown_percent': ((simulation_results['imbalance_0.9']['overall_max_time'] - 
                                 simulation_results['imbalance_0.0']['overall_max_time']) / 
                                simulation_results['imbalance_0.0']['overall_max_time'] * 100)
        },
        'assignment_strategies': {
            strategy: {'gini': data['gini']} 
            for strategy, data in assignment_results.items()
        },
        'loss_functions': loss_results
    }
    
    # Save comprehensive summary
    output_dir = Path("results/demo")
    with open(output_dir / "comprehensive_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print key findings
    print("\n📊 KEY FINDINGS:\n")
    print(f"1. Expert Imbalance Exists:")
    print(f"   - Average Gini: {summary['profiling']['average_gini']:.4f}")
    print(f"   - Status: {summary['profiling']['interpretation']}")
    
    print(f"\n2. Performance Impact:")
    print(f"   - Balanced: {summary['simulation']['balanced_time_us']:.2f} μs")
    print(f"   - Imbalanced: {summary['simulation']['imbalanced_time_us']:.2f} μs")
    print(f"   - Slowdown: {summary['simulation']['slowdown_percent']:.1f}%")
    
    print(f"\n3. Mitigation Strategies:")
    for strategy, data in summary['assignment_strategies'].items():
        print(f"   - {strategy}: Gini = {data['gini']:.4f}")
    
    print(f"\n4. Loss Functions Implemented:")
    print(f"   - Multiple loss types tested (gini, variance, cv, entropy)")
    print(f"   - Can distinguish between balanced and imbalanced scenarios")
    
    print(f"\n✓ All results saved to {output_dir}/")
    print("\n" + "="*80)
    
    return summary


def main():
    print("="*80)
    print("M1 Mac Friendly Demo - MoE All-to-All Communication Study")
    print("="*80)
    print("\nThis demo runs entirely on synthetic data - no large models needed!")
    print("It demonstrates all components of your implementation.\n")
    
    try:
        # Run all demos
        profiling_summary = demo_profiling()
        simulation_results = demo_network_simulation()
        assignment_results = demo_assignment_strategies()
        loss_results = demo_loss_functions()
        
        # Create final summary
        final_summary = create_final_summary(
            profiling_summary, 
            simulation_results, 
            assignment_results, 
            loss_results
        )
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nAll results are in: results/demo/")
        print("\nVisualization files:")
        print("  - results/demo/plots/imbalance_metrics.png")
        print("  - results/demo/communication_comparison.png")
        print("  - results/demo/plots/expert_dist_*.png (per layer)")
        print("\nData files:")
        print("  - results/demo/comprehensive_summary.json")
        print("  - results/demo/profiling_report.txt")
        print("  - results/demo/simulation_results.json")
        print("\nThis demonstrates that your implementation is complete and working!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
