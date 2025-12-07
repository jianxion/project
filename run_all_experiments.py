#!/usr/bin/env python3
"""
Main script to run all experiments in sequence.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all MoE experiments")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                       help="Model to use (default: Mixtral which doesn't require flash_attn)")
    parser.add_argument("--skip-profiling", action="store_true",
                       help="Skip profiling experiment")
    parser.add_argument("--skip-assignment", action="store_true",
                       help="Skip assignment test experiment")
    parser.add_argument("--skip-finetuning", action="store_true",
                       help="Skip fine-tuning experiment")
    parser.add_argument("--skip-simulation", action="store_true",
                       help="Skip network simulation")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick version with fewer samples")
    
    args = parser.parse_args()
    
    # Set sample counts based on mode
    max_samples = 50 if args.quick else 100
    train_samples = 100 if args.quick else 1000
    
    print("="*80)
    print("MoE All-to-All Communication Study")
    print("Running All Experiments")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print()
    
    results = {}
    
    # Experiment 1: Profile expert assignments
    if not args.skip_profiling:
        cmd = (f"python experiments/profile_expert_assignment.py "
               f"--model {args.model} "
               f"--max_samples {max_samples}")
        results['profiling'] = run_command(cmd, "Experiment 1: Expert Assignment Profiling")
    
    # Experiment 2: Test assignment modifications
    if not args.skip_assignment:
        cmd = (f"python experiments/test_assignment_changes.py "
               f"--model {args.model} "
               f"--max_samples {max_samples}")
        results['assignment'] = run_command(cmd, "Experiment 2: Assignment Modification Tests")
    
    # Experiment 3: Fine-tune with imbalance loss
    if not args.skip_finetuning:
        cmd = (f"python experiments/finetune_with_imbalance_loss.py "
               f"--model {args.model} "
               f"--max_samples {train_samples} "
               f"--num_epochs 1")  # Use 1 epoch for quick testing
        results['finetuning'] = run_command(cmd, "Experiment 3: Fine-tuning with Imbalance Loss")
    
    # Experiment 4: Network simulation
    if not args.skip_simulation:
        cmd = "python experiments/simulate_network.py --num_tokens 1024"
        results['simulation'] = run_command(cmd, "Experiment 4: Network Performance Simulation")
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    
    for experiment, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{experiment.capitalize():<30} {status}")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    print("\nResults can be found in:")
    print("  - results/profiling/         : Expert assignment profiles")
    print("  - results/assignment_tests/  : Assignment modification results")
    print("  - results/finetuned_model/   : Fine-tuned model checkpoints")
    print("  - results/simulation/        : Network simulation results")
    print("\nVisualization plots:")
    print("  - results/profiling/plots/")
    print("  - results/simulation/")
    print()
    
    # Check if all succeeded
    if all(results.values()):
        print("✓ All experiments completed successfully!")
        return 0
    else:
        print("⚠ Some experiments failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
