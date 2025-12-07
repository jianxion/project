"""
Perplexity evaluation for language models on benchmark datasets.
"""

# Mock flash_attn if not available
import sys
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
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import json
from pathlib import Path


class PerplexityEvaluator:
    """Evaluate language model perplexity on benchmark datasets."""
    
    def __init__(self, 
                 model_name_or_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize evaluator.
        
        Args:
            model_name_or_path: Model name or path to checkpoint
            device: Device to run evaluation on
        """
        self.device = device
        print(f"Loading model from {model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try to load with attn_implementation parameter, fallback if not supported
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager attention instead of flash_attn
                _attn_implementation="eager"
            )
        except (TypeError, ImportError) as e:
            print(f"Note: Loading with fallback parameters due to: {e}")
            # Fallback: load without attn_implementation
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()
        
    def compute_perplexity(self, 
                          texts: List[str], 
                          max_length: int = 512,
                          batch_size: int = 8) -> Dict:
        """
        Compute perplexity on a list of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with perplexity metrics
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # Accumulate loss (weighted by number of tokens)
                loss = outputs.loss
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': float(perplexity),
            'avg_loss': float(avg_loss),
            'total_tokens': int(total_tokens),
            'num_samples': len(texts)
        }
    
    def evaluate_wikitext(self, 
                         split: str = "test",
                         max_samples: Optional[int] = None,
                         max_length: int = 512) -> Dict:
        """
        Evaluate on WikiText-2 dataset.
        
        Args:
            split: Dataset split ('test', 'validation')
            max_samples: Maximum number of samples (None = all)
            max_length: Maximum sequence length
            
        Returns:
            Perplexity results
        """
        print(f"Loading WikiText-2 {split} set")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # Filter empty texts
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
        
        if max_samples:
            texts = texts[:max_samples]
        
        print(f"Evaluating on {len(texts)} samples")
        results = self.compute_perplexity(texts, max_length=max_length)
        results['dataset'] = 'wikitext-2'
        results['split'] = split
        
        return results
    
    def evaluate_c4(self,
                   split: str = "validation",
                   max_samples: int = 1000,
                   max_length: int = 512) -> Dict:
        """
        Evaluate on C4 (Colossal Clean Crawled Corpus) dataset.
        
        Args:
            split: Dataset split ('validation')
            max_samples: Maximum number of samples
            max_length: Maximum sequence length
            
        Returns:
            Perplexity results
        """
        print(f"Loading C4 {split} set")
        dataset = load_dataset("c4", "en", split=split, streaming=True)
        
        # Take first max_samples
        texts = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            texts.append(item['text'])
        
        print(f"Evaluating on {len(texts)} samples")
        results = self.compute_perplexity(texts, max_length=max_length)
        results['dataset'] = 'c4'
        results['split'] = split
        
        return results
    
    def evaluate_lambada(self,
                        max_samples: Optional[int] = None,
                        max_length: int = 512) -> Dict:
        """
        Evaluate on LAMBADA dataset (word prediction task).
        
        Args:
            max_samples: Maximum number of samples
            max_length: Maximum sequence length
            
        Returns:
            Perplexity results
        """
        print("Loading LAMBADA dataset")
        dataset = load_dataset("lambada", split="test")
        
        texts = [item['text'] for item in dataset]
        
        if max_samples:
            texts = texts[:max_samples]
        
        print(f"Evaluating on {len(texts)} samples")
        results = self.compute_perplexity(texts, max_length=max_length)
        results['dataset'] = 'lambada'
        results['split'] = 'test'
        
        return results
    
    def run_full_evaluation(self, 
                           output_path: str = "results/perplexity_results.json") -> Dict:
        """
        Run evaluation on all benchmark datasets.
        
        Args:
            output_path: Path to save results
            
        Returns:
            Combined results dictionary
        """
        results = {}
        
        # WikiText-2
        try:
            print("\n" + "="*80)
            print("Evaluating on WikiText-2")
            print("="*80)
            results['wikitext'] = self.evaluate_wikitext()
            print(f"WikiText-2 Perplexity: {results['wikitext']['perplexity']:.2f}")
        except Exception as e:
            print(f"Error evaluating WikiText-2: {e}")
            results['wikitext'] = {'error': str(e)}
        
        # C4
        try:
            print("\n" + "="*80)
            print("Evaluating on C4")
            print("="*80)
            results['c4'] = self.evaluate_c4()
            print(f"C4 Perplexity: {results['c4']['perplexity']:.2f}")
        except Exception as e:
            print(f"Error evaluating C4: {e}")
            results['c4'] = {'error': str(e)}
        
        # LAMBADA
        try:
            print("\n" + "="*80)
            print("Evaluating on LAMBADA")
            print("="*80)
            results['lambada'] = self.evaluate_lambada()
            print(f"LAMBADA Perplexity: {results['lambada']['perplexity']:.2f}")
        except Exception as e:
            print(f"Error evaluating LAMBADA: {e}")
            results['lambada'] = {'error': str(e)}
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        return results


def compare_perplexity_results(baseline_results: Dict, 
                               modified_results: Dict,
                               output_path: str = "results/perplexity_comparison.json") -> Dict:
    """
    Compare perplexity results between baseline and modified models.
    
    Args:
        baseline_results: Results from baseline model
        modified_results: Results from modified model
        output_path: Path to save comparison
        
    Returns:
        Comparison dictionary
    """
    comparison = {}
    
    for dataset in ['wikitext', 'c4', 'lambada']:
        if dataset in baseline_results and dataset in modified_results:
            baseline_ppl = baseline_results[dataset].get('perplexity', None)
            modified_ppl = modified_results[dataset].get('perplexity', None)
            
            if baseline_ppl and modified_ppl:
                ppl_change = ((modified_ppl - baseline_ppl) / baseline_ppl) * 100
                
                comparison[dataset] = {
                    'baseline_perplexity': baseline_ppl,
                    'modified_perplexity': modified_ppl,
                    'absolute_change': modified_ppl - baseline_ppl,
                    'relative_change_percent': ppl_change
                }
    
    # Save comparison
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to {output_path}")
    
    return comparison


if __name__ == "__main__":
    # Example usage
    evaluator = PerplexityEvaluator("microsoft/Phi-3.5-MoE-instruct")
    results = evaluator.run_full_evaluation()
    
    print("\nSummary:")
    for dataset, metrics in results.items():
        if 'perplexity' in metrics:
            print(f"{dataset}: {metrics['perplexity']:.2f}")
