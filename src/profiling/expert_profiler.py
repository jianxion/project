"""
Expert assignment profiler for MoE models.
Captures and analyzes expert routing decisions during inference.
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
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path


class ExpertAssignmentProfiler:
    """Profiles expert assignments in MoE models."""
    
    def __init__(self, model_name: str = "microsoft/Phi-3.5-MoE-instruct"):
        """
        Initialize the profiler.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.expert_assignments = defaultdict(list)
        self.routing_weights = defaultdict(list)
        self.hooks = []
        
    def load_model(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load the MoE model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Try to load with attn_implementation parameter, fallback if not supported
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
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
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()
        print(f"Model loaded on {device}")
        
    def register_hooks(self):
        """Register forward hooks to capture expert assignments."""
        self.expert_assignments.clear()
        self.routing_weights.clear()
        
        def hook_fn(name):
            def hook(module, input, output):
                # Capture routing decisions
                # This will vary based on the specific MoE implementation
                if hasattr(module, 'router'):
                    router_output = module.router(input[0])
                    # Store top-k expert indices
                    if isinstance(router_output, tuple):
                        routing_weights, expert_indices = router_output
                    else:
                        routing_weights = router_output
                        expert_indices = torch.topk(routing_weights, k=module.num_experts_per_tok, dim=-1).indices
                    
                    self.expert_assignments[name].append(expert_indices.detach().cpu().numpy())
                    self.routing_weights[name].append(routing_weights.detach().cpu().numpy())
            return hook
        
        # Register hooks on MoE layers
        for name, module in self.model.named_modules():
            if 'block_sparse_moe' in name.lower() or 'moe' in name.lower():
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
                print(f"Registered hook on: {name}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        
    def profile_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Profile expert assignments for a given text.
        
        Args:
            text: Input text to profile
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with profiling statistics
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        return self.compute_statistics()
    
    def profile_dataset(self, dataset: List[str], max_samples: int = 100) -> Dict:
        """
        Profile expert assignments across a dataset.
        
        Args:
            dataset: List of text samples
            max_samples: Maximum number of samples to profile
            
        Returns:
            Aggregated profiling statistics
        """
        self.register_hooks()
        
        for i, text in enumerate(dataset[:max_samples]):
            if i % 10 == 0:
                print(f"Profiling sample {i}/{min(max_samples, len(dataset))}")
            
            self.profile_text(text)
        
        stats = self.compute_statistics()
        self.remove_hooks()
        
        return stats
    
    def compute_statistics(self) -> Dict:
        """
        Compute statistics from collected expert assignments.
        
        Returns:
            Dictionary with various imbalance metrics
        """
        stats = {}
        
        for layer_name, assignments in self.expert_assignments.items():
            if not assignments:
                continue
                
            # Concatenate all assignments for this layer
            all_assignments = np.concatenate(assignments, axis=0)
            
            # Flatten to get expert frequency distribution
            expert_counts = np.bincount(all_assignments.flatten())
            total_assignments = len(all_assignments.flatten())
            
            # Calculate imbalance metrics
            expert_probs = expert_counts / total_assignments
            num_experts = len(expert_counts)
            
            # Gini coefficient (0 = perfect balance, 1 = complete imbalance)
            sorted_probs = np.sort(expert_probs)
            cumsum = np.cumsum(sorted_probs)
            gini = (num_experts + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / num_experts if cumsum[-1] > 0 else 0
            
            # Standard deviation from uniform distribution
            uniform_prob = 1.0 / num_experts
            std_dev = np.std(expert_probs)
            
            # Max/min ratio
            max_min_ratio = expert_probs.max() / (expert_probs.min() + 1e-10)
            
            # Coefficient of variation
            cv = std_dev / (expert_probs.mean() + 1e-10)
            
            stats[layer_name] = {
                'expert_counts': expert_counts.tolist(),
                'expert_probs': expert_probs.tolist(),
                'gini_coefficient': float(gini),
                'std_deviation': float(std_dev),
                'max_min_ratio': float(max_min_ratio),
                'coefficient_of_variation': float(cv),
                'total_assignments': int(total_assignments),
                'num_experts': int(num_experts)
            }
        
        return stats
    
    def save_statistics(self, stats: Dict, output_path: str):
        """Save statistics to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {output_path}")
    
    def get_imbalance_summary(self, stats: Dict) -> Dict:
        """
        Get a summary of imbalance across all layers.
        
        Returns:
            Summary statistics
        """
        gini_coeffs = [s['gini_coefficient'] for s in stats.values()]
        cvs = [s['coefficient_of_variation'] for s in stats.values()]
        
        summary = {
            'average_gini': float(np.mean(gini_coeffs)),
            'max_gini': float(np.max(gini_coeffs)),
            'average_cv': float(np.mean(cvs)),
            'max_cv': float(np.max(cvs)),
            'num_layers': len(stats)
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    profiler = ExpertAssignmentProfiler()
    profiler.load_model()
    
    # Profile on sample text
    sample_text = "The quick brown fox jumps over the lazy dog."
    profiler.register_hooks()
    stats = profiler.profile_text(sample_text)
    profiler.remove_hooks()
    
    profiler.save_statistics(stats, "results/expert_assignment_stats.json")
    
    summary = profiler.get_imbalance_summary(stats)
    print("\nImbalance Summary:")
    print(json.dumps(summary, indent=2))
