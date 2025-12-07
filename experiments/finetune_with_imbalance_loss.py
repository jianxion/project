#!/usr/bin/env python3
"""
Experiment 3: Fine-tune MoE model with expert imbalance loss.
"""

import sys
sys.path.append('.')

# Mock flash_attn if not available
try:
    import flash_attn
except ImportError:
    # Create mock flash_attn module
    from types import ModuleType
    flash_attn = ModuleType('flash_attn')
    sys.modules['flash_attn'] = flash_attn

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.training.imbalance_loss import MoEModelWithImbalanceLoss
from src.training.trainer import MoETrainer
from src.evaluation.perplexity import PerplexityEvaluator
import argparse
from pathlib import Path


def prepare_dataset(tokenizer, dataset_name="wikitext", max_samples=1000, max_length=512):
    """Prepare dataset for training."""
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    else:
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        texts = [item['text'] for i, item in enumerate(dataset) if i < max_samples]
    
    # Tokenize
    encodings = tokenizer(
        texts[:max_samples],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings['input_ids'])
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['input_ids'][idx]
            }
    
    return TextDataset(encodings)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MoE with imbalance loss")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-MoE-instruct",
                       help="Model name or path")
    parser.add_argument("--num_experts", type=int, default=16,
                       help="Number of experts per layer")
    parser.add_argument("--imbalance_weight", type=float, default=0.01,
                       help="Weight for imbalance loss")
    parser.add_argument("--load_balance_weight", type=float, default=0.01,
                       help="Weight for load balancing loss")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum training samples")
    parser.add_argument("--output_dir", type=str, default="results/finetuned_model",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Fine-tuning MoE with Expert Imbalance Loss")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Imbalance weight: {args.imbalance_weight}")
    print(f"Load balance weight: {args.load_balance_weight}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load with attn_implementation parameter, fallback if not supported
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention instead of flash_attn
            _attn_implementation="eager"
        )
    except (TypeError, ImportError) as e:
        print(f"Note: Loading with fallback parameters due to: {e}")
        # Fallback: load without attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Wrap model with imbalance loss
    print("Wrapping model with imbalance loss...")
    model = MoEModelWithImbalanceLoss(
        base_model,
        num_experts=args.num_experts,
        imbalance_weight=args.imbalance_weight,
        load_balance_weight=args.load_balance_weight
    )
    model.cache_router_logits()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = prepare_dataset(tokenizer, max_samples=args.max_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optional: prepare validation set
    eval_dataset = prepare_dataset(tokenizer, max_samples=100)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = MoETrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir
    )
    
    # Evaluate baseline
    print("\nEvaluating baseline model...")
    evaluator = PerplexityEvaluator(args.model)
    baseline_results = evaluator.evaluate_wikitext(max_samples=100)
    print(f"Baseline Perplexity: {baseline_results['perplexity']:.2f}")
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    # Note: In practice, you'd load the fine-tuned checkpoint
    # For now, this demonstrates the workflow
    print("⚠️  Note: Full evaluation requires loading the fine-tuned checkpoint")
    print("    See the training_history.json for loss curves")
    
    print("\n" + "="*80)
    print("Fine-tuning Complete!")
    print("="*80)
    print(f"Checkpoints saved to: {args.output_dir}/")
    print(f"Training history: {args.output_dir}/training_history.json")
    print()
    print("Next steps:")
    print("1. Check training_history.json for loss curves")
    print("2. Evaluate fine-tuned model perplexity")
    print("3. Profile expert assignments in fine-tuned model")
    print("4. Compare against baseline")


if __name__ == "__main__":
    main()
