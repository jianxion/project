#!/usr/bin/env python3
"""
Diagnostic version: Evaluate on BOTH train and test sets to show generalization.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from demo_lightweight_real import profile_moe_model, get_sample_texts
from src.profiling.accuracy_evaluator import SimpleAccuracyEvaluator


class BalanceLoss(nn.Module):
    """Simple balance loss to encourage uniform expert usage."""
    
    def __init__(self, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.uniform_target = 1.0 / num_experts
    
    def forward(self, router_logits):
        """Calculate variance-based balance loss."""
        probs = torch.softmax(router_logits, dim=-1)
        avg_usage = probs.mean(dim=0)
        target = torch.full_like(avg_usage, self.uniform_target)
        loss = ((avg_usage - target) ** 2).mean()
        return loss


def joint_training_with_balance_loss(
    model,
    tokenizer,
    train_texts,
    num_steps=40,
    balance_weight=1.0,
    learning_rate=5e-6
):
    """Train routers AND experts together with balance loss."""
    print("\n" + "="*80)
    print("JOINT TRAINING: Routers + Experts with Balance Loss")
    print("="*80)
    print(f"Steps: {num_steps}")
    print(f"Balance weight: {balance_weight}")
    print(f"Learning rate: {learning_rate}")
    print("="*80 + "\n")
    
    model.train()
    
    # Select parameters to train
    trainable_params = []
    param_names = []
    
    for name, param in model.named_parameters():
        if 'router.classifier' in name:
            param.requires_grad = True
            trainable_params.append(param)
            param_names.append(name)
        elif 'mlp.experts.expert' in name and ('wi.weight' in name or 'wi_0.weight' in name):
            param.requires_grad = True
            trainable_params.append(param)
            param_names.append(name)
        else:
            param.requires_grad = False
    
    print(f"Training {len(trainable_params)} parameter groups:")
    print(f"  - {sum(1 for n in param_names if 'router' in n)} router layers")
    print(f"  - {sum(1 for n in param_names if 'expert' in n)} expert layers")
    print(f"  - Total params: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M\n")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    balance_loss_fn = BalanceLoss(num_experts=8)
    
    # Track training
    history = {
        'step': [],
        'total_loss': [],
        'model_loss': [],
        'balance_loss': [],
        'avg_gini': []
    }
    
    # Training loop
    print("Starting training...")
    for step in tqdm(range(num_steps), desc="Training"):
        text = train_texts[step % len(train_texts)]
        
        inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding=True)
        if 'decoder_input_ids' not in inputs:
            inputs['decoder_input_ids'] = inputs['input_ids'].clone()
        inputs['labels'] = inputs['input_ids'].clone()
        
        # Collect router outputs
        router_outputs = []
        hooks = []
        
        def capture_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                router_outputs.append(output)
            elif isinstance(output, tuple):
                router_outputs.append(output[0])
        
        for name, module in model.named_modules():
            if 'router.classifier' in name:
                hooks.append(module.register_forward_hook(capture_hook))
        
        # Forward
        outputs = model(**inputs)
        model_loss = outputs.loss
        
        # Balance loss
        total_balance_loss = 0
        gini_coeffs = []
        
        for router_logits in router_outputs:
            if router_logits.numel() > 0:
                if len(router_logits.shape) == 3:
                    router_logits = router_logits.view(-1, router_logits.shape[-1])
                
                balance_loss = balance_loss_fn(router_logits)
                total_balance_loss += balance_loss
                
                with torch.no_grad():
                    probs = torch.softmax(router_logits, dim=-1).mean(dim=0).cpu().numpy()
                    probs = np.abs(probs) + 1e-10
                    probs = probs / probs.sum()
                    sorted_probs = np.sort(probs)
                    n = len(sorted_probs)
                    gini = (n + 1 - 2 * np.sum(np.cumsum(sorted_probs)) / np.sum(sorted_probs)) / n
                    gini = max(0.0, min(1.0, gini))
                    gini_coeffs.append(gini)
        
        for hook in hooks:
            hook.remove()
        
        avg_balance_loss = total_balance_loss / len(router_outputs) if router_outputs else 0
        total_loss = model_loss + balance_weight * avg_balance_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        # Record
        avg_gini = np.mean(gini_coeffs) if gini_coeffs else 0
        history['step'].append(step + 1)
        history['total_loss'].append(total_loss.item())
        history['model_loss'].append(model_loss.item())
        history['balance_loss'].append(avg_balance_loss.item() if isinstance(avg_balance_loss, torch.Tensor) else avg_balance_loss)
        history['avg_gini'].append(avg_gini)
        
        if (step + 1) % 10 == 0:
            print(f"\nStep {step+1}/{num_steps}:")
            print(f"  Model loss: {model_loss.item():.4f}")
            print(f"  Balance loss: {avg_balance_loss.item():.4f}")
            print(f"  Avg Gini: {avg_gini:.4f}")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    return history


def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC: Train vs Test Set Evaluation")
    print("="*80)
    
    # Load model
    model_name = "google/switch-base-8"
    print(f"\nLoading: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.to("cpu")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded\n")
    
    # Get data
    texts = get_sample_texts()
    train_texts = texts[:15]
    test_texts = texts[15:]
    
    print(f"Dataset: {len(train_texts)} train, {len(test_texts)} test")
    
    # Evaluate BEFORE on both sets
    print("\n" + "="*80)
    print("BEFORE TRAINING")
    print("="*80)
    
    model.eval()
    
    print("\n[TRAIN SET]")
    train_stats_before = profile_moe_model(model, tokenizer, train_texts, num_experts=8)
    train_gini_before = np.mean([s['gini_coefficient'] for s in train_stats_before.values()])
    print(f"Train Gini: {train_gini_before:.4f}")
    
    print("\n[TEST SET]")
    test_stats_before = profile_moe_model(model, tokenizer, test_texts, num_experts=8)
    test_gini_before = np.mean([s['gini_coefficient'] for s in test_stats_before.values()])
    print(f"Test Gini: {test_gini_before:.4f}")
    
    print("\n[ACCURACY BASELINE]")
    evaluator = SimpleAccuracyEvaluator(model, tokenizer)
    accuracy_before = evaluator.quick_accuracy_check(test_texts[:10])
    
    # TRAIN
    print("\n" + "="*80)
    balance_weight = float(input("Enter balance_weight (try 500.0 or more): "))
    
    training_history = joint_training_with_balance_loss(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        num_steps=40,
        balance_weight=balance_weight,
        learning_rate=5e-6
    )
    
    # Evaluate AFTER on both sets
    print("\n" + "="*80)
    print("AFTER TRAINING")
    print("="*80)
    
    model.eval()
    
    print("\n[TRAIN SET]")
    train_stats_after = profile_moe_model(model, tokenizer, train_texts, num_experts=8)
    train_gini_after = np.mean([s['gini_coefficient'] for s in train_stats_after.values()])
    print(f"Train Gini: {train_gini_after:.4f}")
    
    print("\n[TEST SET]")
    test_stats_after = profile_moe_model(model, tokenizer, test_texts, num_experts=8)
    test_gini_after = np.mean([s['gini_coefficient'] for s in test_stats_after.values()])
    print(f"Test Gini: {test_gini_after:.4f}")
    
    # Measure accuracy impact
    print("\n[ACCURACY IMPACT]")
    evaluator = SimpleAccuracyEvaluator(model, tokenizer)
    accuracy_after = evaluator.quick_accuracy_check(test_texts[:10])
    
    # Compare
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    train_improvement = (train_gini_before - train_gini_after) / train_gini_before * 100
    test_improvement = (test_gini_before - test_gini_after) / test_gini_before * 100
    
    print(f"\nBALANCE:")
    print(f"  Train: {train_gini_before:.4f} → {train_gini_after:.4f} ({train_improvement:+.1f}%)")
    print(f"  Test:  {test_gini_before:.4f} → {test_gini_after:.4f} ({test_improvement:+.1f}%)")
    
    print(f"\nACCURACY (Test Set Perplexity):")
    print(f"  Before: {accuracy_before['perplexity']:.2f}")
    print(f"  After:  {accuracy_after['perplexity']:.2f}")
    ppl_change = (accuracy_after['perplexity'] - accuracy_before['perplexity']) / accuracy_before['perplexity'] * 100
    print(f"  Change: {ppl_change:+.1f}%")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHT:")
    print("="*80)
    if train_improvement > 3 and test_improvement > 3:
        print("✓ SUCCESS! Training worked and generalized to unseen data")
        print(f"  → Balance improved ~{test_improvement:.0f}% on new texts")
        if abs(ppl_change) < 50:
            print(f"  → Accuracy preserved (perplexity change {ppl_change:+.0f}%)")
        else:
            print(f"  → Accuracy degraded significantly ({ppl_change:+.0f}%)")
        print("\n  This proves joint training can improve balance without")
        print("  catastrophic accuracy loss (unlike naive methods).")
    elif train_improvement > 3 and test_improvement < 2:
        print("✓ Training worked on train set but didn't generalize to test set")
        print("  → This is expected with only 15 training samples")
        print("  → For real deployment, need larger diverse dataset")
    else:
        print("✗ Minimal improvement - balance weight still too low")
        print(f"  → Try balance_weight={balance_weight*5:.0f} or train longer")
    
    # Save
    output_dir = Path("results/joint_training_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'balance_weight': float(balance_weight),
        'train_set': {
            'gini_before': float(train_gini_before),
            'gini_after': float(train_gini_after),
            'improvement_pct': float(train_improvement)
        },
        'test_set': {
            'gini_before': float(test_gini_before),
            'gini_after': float(test_gini_after),
            'improvement_pct': float(test_improvement)
        },
        'training_history': {
            'final_gini': float(training_history['avg_gini'][-1]) if training_history['avg_gini'] else None
        }
    }
    
    with open(output_dir / f"results_weight_{int(balance_weight)}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
