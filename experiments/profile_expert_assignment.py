#!/usr/bin/env python3
"""
Experiment 1: Profile expert assignment distribution in MoE model.
"""

import sys
sys.path.append('.')

from src.profiling.expert_profiler import ExpertAssignmentProfiler
from src.profiling.visualization import (
    plot_expert_distribution, 
    plot_imbalance_metrics,
    plot_heatmap_across_layers,
    create_summary_report
)
from datasets import load_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description="Profile expert assignments in MoE model")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-MoE-instruct",
                       help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "c4", "custom"],
                       help="Dataset to profile on")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to profile")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="results/profiling",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Expert Assignment Profiling Experiment")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print()
    
    # Initialize profiler
    profiler = ExpertAssignmentProfiler(args.model)
    profiler.load_model()
    
    # Load dataset
    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    elif args.dataset == "c4":
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
        texts = [item['text'] for i, item in enumerate(dataset) if i < args.max_samples]
    else:
        # Use sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large amounts of data.",
        ] * (args.max_samples // 3)
    
    # Profile expert assignments
    print(f"\nProfiling {min(args.max_samples, len(texts))} samples...")
    stats = profiler.profile_dataset(texts[:args.max_samples], max_samples=args.max_samples)
    
    # Get summary
    summary = profiler.get_imbalance_summary(stats)
    
    # Save statistics
    profiler.save_statistics(stats, f"{args.output_dir}/expert_assignment_stats.json")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_expert_distribution(stats, output_dir=f"{args.output_dir}/plots")
    plot_imbalance_metrics(stats, output_path=f"{args.output_dir}/plots/imbalance_metrics.png")
    plot_heatmap_across_layers(stats, output_path=f"{args.output_dir}/plots/expert_heatmap.png")
    
    # Create summary report
    create_summary_report(stats, summary, output_path=f"{args.output_dir}/profiling_report.txt")
    
    # Print summary
    print("\n" + "="*80)
    print("Profiling Summary")
    print("="*80)
    print(f"Number of MoE layers: {summary['num_layers']}")
    print(f"Average Gini coefficient: {summary['average_gini']:.4f}")
    print(f"Maximum Gini coefficient: {summary['max_gini']:.4f}")
    print(f"Average CV: {summary['average_cv']:.4f}")
    print(f"Maximum CV: {summary['max_cv']:.4f}")
    print()
    
    if summary['average_gini'] > 0.2:
        print("⚠️  HIGH IMBALANCE DETECTED: Expert assignment shows significant imbalance")
    elif summary['average_gini'] > 0.1:
        print("⚠️  MODERATE IMBALANCE: Expert assignment shows some imbalance")
    else:
        print("✓ BALANCED: Expert assignment is relatively balanced")
    
    print(f"\nResults saved to {args.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
