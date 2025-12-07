#!/usr/bin/env python3
"""
M1-friendly demo using GPT-2 to demonstrate expert routing concepts.
Uses attention patterns as a proxy for expert assignments.
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time

from src.profiling.visualization import (
    plot_expert_distribution,
    plot_imbalance_metrics,
    create_summary_report
)


def get_sample_texts():
    """Get diverse sample texts."""
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
        "Biometric authentication enhances security systems.",
        # Add more varied content
        "The history of mathematics dates back thousands of years.",
        "Photosynthesis is the process by which plants make food.",
        "The human brain contains approximately 86 billion neurons.",
        "Democracy requires active participation from citizens.",
        "Musical harmony arises from specific frequency relationships.",
        "Economic systems allocate resources among competing needs.",
        "Chemical reactions involve the rearrangement of atoms.",
        "Literature reflects and shapes cultural values.",
        "Protein folding determines biological function.",
        "Urban planning balances growth with sustainability."
    ]


def analyze_attention_as_expert_routing(model, tokenizer, texts, num_experts=8, device='cpu'):
    """
    Analyze attention patterns and map them to simulated expert routing.
    This demonstrates how token specialization works in transformers.
    """
    print(f"\n{'='*80}")
    print("Analyzing Attention Patterns as Expert Routing Proxy")
    print(f"{'='*80}")
    print("\nConcept: Different tokens pay attention to different parts of context.")
    print("We simulate 8 'experts' by grouping attention patterns.\n")
    
    model.eval()
    
    # Collect attention patterns for each layer
    layer_attention_patterns = {f"layer_{i}": [] for i in range(model.config.num_hidden_layers)}
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing texts"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, output_attentions=True)
            
            # Process each layer's attention
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                # layer_attn shape: [batch, num_heads, seq_len, seq_len]
                # Average over batch and heads
                avg_attn = layer_attn.mean(dim=(0, 1)).cpu().numpy()  # [seq_len, seq_len]
                
                layer_attention_patterns[f"layer_{layer_idx}"].append(avg_attn)
    
    # Convert attention patterns to expert assignments
    stats = {}
    
    for layer_idx in range(min(8, model.config.num_hidden_layers)):  # Analyze first 8 layers
        layer_key = f"layer_{layer_idx}"
        attention_matrices = layer_attention_patterns[layer_key]
        
        expert_counts = np.zeros(num_experts)
        
        for attn_matrix in attention_matrices:
            seq_len = attn_matrix.shape[0]
            
            for token_idx in range(seq_len):
                # Each token's attention pattern determines its "expert"
                token_attn = attn_matrix[token_idx]
                
                # Method 1: Cluster based on where attention focuses
                # High attention to self -> Expert 0 (local expert)
                # High attention to first tokens -> Expert 1 (context expert)
                # High attention to recent tokens -> Expert 2 (recency expert)
                # Distributed attention -> Expert 3-7 (various specializations)
                
                self_attn = token_attn[token_idx] if token_idx < len(token_attn) else 0
                start_attn = token_attn[:max(1, token_idx//4)].sum() if token_idx > 0 else 0
                recent_attn = token_attn[max(0, token_idx-3):token_idx].sum() if token_idx > 0 else 0
                
                # Assign to expert based on attention pattern
                attn_scores = [
                    self_attn * 2,  # Expert 0: self-attention
                    start_attn,     # Expert 1: beginning context
                    recent_attn,    # Expert 2: recent context
                ]
                
                # Add more experts based on attention distribution
                attention_entropy = -np.sum(token_attn * np.log(token_attn + 1e-10))
                uniform_bins = np.array_split(token_attn, num_experts - 3)
                for bin_attn in uniform_bins:
                    attn_scores.append(bin_attn.sum())
                
                # Top-2 routing (like MoE)
                top_2_experts = np.argsort(attn_scores)[-2:]
                for expert_id in top_2_experts:
                    if expert_id < num_experts:
                        expert_counts[expert_id] += 1
        
        # Calculate statistics
        if expert_counts.sum() > 0:
            probs = expert_counts / expert_counts.sum()
            sorted_probs = np.sort(probs)
            n = len(probs)
            gini = (n + 1 - 2 * np.sum(np.cumsum(sorted_probs)) / np.sum(sorted_probs)) / n
            
            std_dev = float(np.std(probs))
            cv = std_dev / (probs.mean() + 1e-10)
            max_min = probs.max() / (probs.min() + 1e-10)
            
            stats[f"model.layers.{layer_idx}.attention_routing"] = {
                'expert_counts': expert_counts.tolist(),
                'expert_probs': probs.tolist(),
                'gini_coefficient': float(gini),
                'std_deviation': std_dev,
                'max_min_ratio': float(max_min),
                'coefficient_of_variation': float(cv),
                'total_assignments': int(expert_counts.sum()),
                'num_experts': num_experts
            }
    
    return stats


def main():
    print("="*80)
    print("GPT-2 ATTENTION-BASED EXPERT ROUTING DEMO")
    print("="*80)
    print("\n📚 Educational Note:")
    print("While GPT-2 doesn't have MoE architecture, we can demonstrate the")
    print("concept by analyzing its attention patterns. Different tokens attend")
    print("to different contexts, similar to how MoE routers assign tokens to")
    print("different experts based on their content.\n")
    
    # Load GPT-2 (small, reliable, M1-compatible)
    model_name = "gpt2"
    print(f"🚀 Loading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        tokenizer.pad_token = tokenizer.eos_token
        device = "cpu"
        model.to(device)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Layers: {model.config.num_hidden_layers}")
        print(f"✓ Attention heads: {model.config.num_attention_heads}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Get sample texts
    texts = get_sample_texts()
    print(f"\n📝 Processing {len(texts)} sample texts...")
    
    # Analyze attention as routing
    start_time = time.time()
    stats = analyze_attention_as_expert_routing(model, tokenizer, texts, num_experts=8, device=device)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Analysis completed in {elapsed:.1f} seconds")
    
    # Save results
    output_dir = Path("results/gpt2_proxy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "attention_routing_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Calculate summary
    gini_coeffs = [s['gini_coefficient'] for s in stats.values()]
    cvs = [s['coefficient_of_variation'] for s in stats.values()]
    
    summary = {
        'model_name': model_name,
        'analysis_type': 'Attention-Based Expert Routing Proxy',
        'num_samples': len(texts),
        'average_gini': float(np.mean(gini_coeffs)),
        'max_gini': float(np.max(gini_coeffs)),
        'min_gini': float(np.min(gini_coeffs)),
        'average_cv': float(np.mean(cvs)),
        'num_layers_analyzed': len(stats)
    }
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    plot_expert_distribution(stats, output_dir=str(output_dir / "plots"))
    plot_imbalance_metrics(stats, output_path=str(output_dir / "plots" / "imbalance_metrics.png"))
    create_summary_report(stats, summary, output_path=str(output_dir / "profiling_report.txt"))
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS - ATTENTION-BASED ROUTING ANALYSIS")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Layers analyzed: {len(stats)}")
    print(f"\n📊 Simulated Expert Imbalance Metrics:")
    print(f"  Average Gini coefficient: {summary['average_gini']:.4f}")
    print(f"  Range: {summary['min_gini']:.4f} - {summary['max_gini']:.4f}")
    print(f"  Average CV: {summary['average_cv']:.4f}")
    
    print(f"\n💡 Interpretation:")
    print(f"  Gini = {summary['average_gini']:.4f} indicates ")
    if summary['average_gini'] > 0.3:
        print("  ⚠️  SIGNIFICANT IMBALANCE in attention patterns")
        print("  → Different tokens specialize in different context patterns")
        print("  → This mirrors how MoE experts specialize in different tasks")
    elif summary['average_gini'] > 0.15:
        print("  ⚠️  MODERATE IMBALANCE in attention patterns")
        print("  → Some attention patterns are more common than others")
    else:
        print("  ✓ RELATIVELY UNIFORM attention patterns")
    
    print(f"\n🎓 For Your Project:")
    print("  While this uses GPT-2 (not a real MoE), it demonstrates the CONCEPT:")
    print("  • Tokens have different specializations (attention patterns)")
    print("  • Some specializations are more common than others (imbalance)")
    print("  • This imbalance affects communication patterns")
    print("  • Same principles apply to real MoE expert routing")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("\nFiles generated:")
    print("  - attention_routing_stats.json")
    print("  - profiling_report.txt")
    print("  - plots/imbalance_metrics.png")
    print("  - plots/expert_dist_*.png")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETED!")
    print("="*80)
    print("\n💭 What This Shows:")
    print("  Even in standard transformers, tokens don't all need the same")
    print("  information (attention imbalance). In MoE models, this translates")
    print("  to expert imbalance, which causes all-to-all communication issues.")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
