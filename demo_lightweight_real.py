#!/usr/bin/env python3
import sys
sys.path.append('.')

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import time

from src.profiling.visualization import (
    plot_expert_distribution,
    plot_imbalance_metrics,
    create_summary_report
)


def download_and_load_model(model_name):
    """Load Switch Transformer model for CPU profiling."""
    print(f"\n{'='*80}")
    print(f"Loading Model: {model_name}")
    print(f"{'='*80}")
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        print("Downloading model (this may take a few minutes first time)...")
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model = model.to("cpu")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f" Model loaded successfully")
        print(f" Model size: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        
        return model, tokenizer
        
    except Exception as e:
        print(f" Error loading model: {e}")
        return None, None


def get_sample_texts():
    """Get sample texts for profiling."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence research.",
        "Climate change poses significant challenges for future generations.",
        "Quantum computing promises to revolutionize computational capabilities.",
        "The Internet of Things connects devices in unprecedented ways.",
        "Renewable energy sources are becoming increasingly cost-effective.",
        "Artificial neural networks mimic biological brain structures.",
        "Space exploration continues to push technological boundaries.",
        "Blockchain technology enables decentralized digital transactions.",
        "Gene editing tools like CRISPR offer new medical possibilities.",
        "5G networks provide faster wireless communication speeds.",
        "Virtual reality creates immersive digital experiences.",
        "Autonomous vehicles navigate roads using sensor fusion.",
        "Natural language processing enables human-computer interaction.",
        "Edge computing processes data closer to its source.",
        "Cybersecurity threats evolve alongside defensive technologies.",
        "Big data analytics extracts insights from massive datasets.",
        "Cloud computing delivers on-demand computing resources.",
        "Robotics automation transforms manufacturing processes.",
        "Biometric authentication enhances security systems."
    ]


def profile_moe_model(model, tokenizer, texts, num_experts=8):
    """
    Profile actual MoE model by hooking into router outputs.
    """
    print(f"\n{'='*80}")
    print("Profiling MoE Model - Capturing Router Decisions")
    print(f"{'='*80}\n")
    
    model.eval()
    
    # Track expert assignments manually
    expert_counts = {}
    router_layer_names = []
    
    # Register hooks to capture router outputs
    router_outputs = []
    
    def hook_fn(module, input, output):
        """Hook to capture router logits or decisions."""
        # Switch Transformer: router.classifier outputs logits directly
        if isinstance(output, torch.Tensor) and len(output.shape) >= 2:
            router_outputs.append(output.detach().cpu())
        # Some models wrap in tuple
        elif isinstance(output, tuple) and len(output) > 0:
            if isinstance(output[0], torch.Tensor) and len(output[0].shape) >= 2:
                router_outputs.append(output[0].detach().cpu())
    
    # Try to find and hook ONLY router.classifier (not all expert modules)
    hooks = []
    for name, module in model.named_modules():
        # Only hook the router classifier, not individual experts
        if 'router.classifier' in name.lower():
            print(f"Hooking router: {name}")
            hooks.append(module.register_forward_hook(hook_fn))
            router_layer_names.append(name)
            expert_counts[name] = np.zeros(num_experts)
    
    if not hooks:
        print("No router layers detected - this model does not appear to be an MoE model")
        return {}
    
    print(f"\n Found {len(hooks)} router layers")
    
    # Run inference
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing texts"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
            
            # For seq2seq models, add decoder_input_ids
            if 'decoder_input_ids' not in inputs:
                inputs['decoder_input_ids'] = inputs['input_ids']
            
            router_outputs.clear()
            outputs = model(**inputs)
            
            # Process router outputs - each corresponds to a hooked router
            for router_idx, router_logits in enumerate(router_outputs):
                if router_idx >= len(router_layer_names):
                    continue
                    
                router_name = router_layer_names[router_idx]
                
                # Handle different shapes
                if len(router_logits.shape) == 3:
                    # Shape: [batch, seq_len, num_experts]
                    router_logits = router_logits.reshape(-1, router_logits.shape[-1])
                
                if len(router_logits.shape) == 2:
                    # Shape: [num_tokens, num_experts]
                    # Get top-k expert choices per token
                    top_k = min(1, router_logits.shape[-1])  # Switch uses top-1
                    expert_indices = torch.argmax(router_logits, dim=-1)
                    
                    # Count expert usage
                    for expert_id in expert_indices:
                        expert_counts[router_name][expert_id.item()] += 1
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Convert counts to statistics
    stats = {}
    for router_name, counts in expert_counts.items():
        if counts.sum() > 0:
            probs = counts / counts.sum()
            sorted_probs = np.sort(probs)
            n = len(probs)
            gini = (n + 1 - 2 * np.sum(np.cumsum(sorted_probs)) / np.sum(sorted_probs)) / n
            
            # Use cleaner layer name
            layer_key = router_name.replace('.classifier', '').replace('encoder.', '').replace('decoder.', '')
            
            stats[layer_key] = {
                'expert_counts': counts.tolist(),
                'expert_probs': probs.tolist(),
                'gini_coefficient': float(gini),
                'std_deviation': float(np.std(probs)),
                'max_min_ratio': float(probs.max() / (probs.min() + 1e-10)),
                'coefficient_of_variation': float(np.std(probs) / (probs.mean() + 1e-10)),
                'total_assignments': int(counts.sum()),
                'num_experts': num_experts
            }
    
    if not stats:
        print("No router data captured - this model may not have standard MoE routers")
        return {}
    
    print(f"Captured routing data from {len(stats)} MoE layers")
    return stats


def main():
    print("="*80)
    print("SWITCH TRANSFORMER MoE PROFILING")
    print("="*80)
    
    model_name = "google/switch-base-8"
    print(f"\n Loading: {model_name}")
    print("   619M parameters | 8 experts per layer | 12 MoE layers\n")
    
    model, tokenizer = download_and_load_model(model_name)
    
    if model is None:
        print("\n Failed to load model.")
        return 1
    
    # Get sample texts
    texts = get_sample_texts()
    print(f"\n Using {len(texts)} sample texts for profiling")
    
    # Profile the model
    start_time = time.time()
    
    try:
        stats = profile_moe_model(model, tokenizer, texts, num_experts=8)
        elapsed = time.time() - start_time
        
        print(f"\n Profiling completed in {elapsed:.1f} seconds")
        
        # Save results
        output_dir = Path("results/real_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "profiling_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Calculate summary
        gini_coeffs = [s['gini_coefficient'] for s in stats.values()]
        cvs = [s['coefficient_of_variation'] for s in stats.values()]
        
        summary = {
            'model_name': model_name,
            'analysis_type': 'MoE Router Analysis',
            'num_samples': len(texts),
            'average_gini': float(np.mean(gini_coeffs)),
            'max_gini': float(np.max(gini_coeffs)),
            'min_gini': float(np.min(gini_coeffs)),
            'average_cv': float(np.mean(cvs)),
            'max_cv': float(np.max(cvs)),
            'num_layers': len(stats)
        }
        
        # Create visualizations
        print("\n Generating visualizations...")
        plot_expert_distribution(stats, output_dir=str(output_dir / "plots"))
        plot_imbalance_metrics(stats, output_path=str(output_dir / "plots" / "imbalance_metrics.png"))
        create_summary_report(stats, summary, output_path=str(output_dir / "profiling_report.txt"))
        
        # Print summary
        print("\n" + "="*80)
        print("PROFILING RESULTS")
        print("="*80)
        print(f"Layers analyzed: {len(stats)}")
        print(f"\n Imbalance Metrics:")
        print(f"  Average Gini coefficient: {summary['average_gini']:.4f}")
        print(f"  Range: {summary['min_gini']:.4f} - {summary['max_gini']:.4f}")
        print(f"  Average CV: {summary['average_cv']:.4f}")
        
        # Interpretation
        print(f"\n Interpretation:")
        if summary['average_gini'] > 0.3:
            print("    HIGH IMBALANCE DETECTED")
            print("  Expert load is significantly unbalanced!")
        elif summary['average_gini'] > 0.1:
            print("    MODERATE IMBALANCE")
            print("  Some experts are used more than others.")
        else:
            print("   RELATIVELY BALANCED")
            print("  Expert usage is fairly uniform.")
        
        print(f"\n Results saved to: {output_dir}/")
        print("\nFiles generated:")
        print(f"  - profiling_stats.json (raw data)")
        print(f"  - profiling_report.txt (detailed report)")
        print(f"  - plots/imbalance_metrics.png (main visualization)")
        print(f"  - plots/expert_dist_*.png (per-layer distributions)")
        
        print("\n" + "="*80)
        print(" SWITCH TRANSFORMER MoE PROFILING COMPLETED!")
        print("="*80)
        print("\n Results Summary:")
        print("  Analyzed real MoE router decisions from Switch Transformer")
        print(f"  Profiled {len(stats)} MoE layers with 8 experts each")
        print(f"  Demonstrated expert load imbalance (Avg Gini: {summary['average_gini']:.4f})")
        print("  Generated visualizations and detailed report")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
