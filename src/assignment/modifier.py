"""
Heuristic methods for modifying expert assignments.
Tests how changes to expert routing affect model accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Callable, Optional
from enum import Enum


class AssignmentStrategy(Enum):
    """Different strategies for modifying expert assignments."""
    BALANCED = "balanced"  # Force balanced assignment
    ROUND_ROBIN = "round_robin"  # Cycle through experts
    RANDOM = "random"  # Random assignment
    TOP_K_FILTERED = "top_k_filtered"  # Remove least-used experts
    LOAD_AWARE = "load_aware"  # Balance based on current load


class ExpertAssignmentModifier:
    """Modifies expert assignments using various heuristic strategies."""
    
    def __init__(self, num_experts: int, top_k: int = 2):
        """
        Initialize the modifier.
        
        Args:
            num_experts: Total number of experts
            top_k: Number of experts to select per token
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_load = np.zeros(num_experts)
        self.assignment_count = 0
        
    def balanced_assignment(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Force balanced expert assignment by adjusting routing probabilities.
        
        Args:
            router_logits: Original routing logits [batch_size, seq_len, num_experts]
            
        Returns:
            Modified expert indices [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = router_logits.shape
        
        # Convert logits to probabilities
        routing_probs = torch.softmax(router_logits, dim=-1)
        
        # Compute current expert loads
        current_load = routing_probs.sum(dim=(0, 1)).cpu().numpy()
        
        # Penalize overloaded experts
        load_penalty = torch.tensor(current_load / (current_load.mean() + 1e-10), 
                                    device=router_logits.device, dtype=router_logits.dtype)
        
        # Adjust logits to balance load
        adjusted_logits = router_logits - torch.log(load_penalty + 1e-10)
        
        # Select top-k experts from adjusted distribution
        expert_indices = torch.topk(adjusted_logits, k=self.top_k, dim=-1).indices
        
        return expert_indices
    
    def round_robin_assignment(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Assign experts in round-robin fashion.
        
        Args:
            router_logits: Original routing logits [batch_size, seq_len, num_experts]
            
        Returns:
            Modified expert indices [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = router_logits.shape
        
        expert_indices = torch.zeros(batch_size, seq_len, self.top_k, 
                                     dtype=torch.long, device=router_logits.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Assign next top_k experts in round-robin
                start_expert = self.assignment_count % self.num_experts
                experts = [(start_expert + i) % self.num_experts for i in range(self.top_k)]
                expert_indices[b, s] = torch.tensor(experts, device=router_logits.device)
                self.assignment_count += 1
        
        return expert_indices
    
    def random_assignment(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Randomly assign experts.
        
        Args:
            router_logits: Original routing logits [batch_size, seq_len, num_experts]
            
        Returns:
            Modified expert indices [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = router_logits.shape
        
        expert_indices = torch.zeros(batch_size, seq_len, self.top_k, 
                                     dtype=torch.long, device=router_logits.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Random selection without replacement
                experts = torch.randperm(self.num_experts, device=router_logits.device)[:self.top_k]
                expert_indices[b, s] = experts
        
        return expert_indices
    
    def top_k_filtered_assignment(self, router_logits: torch.Tensor, 
                                  excluded_experts: List[int]) -> torch.Tensor:
        """
        Assign experts while excluding certain experts (e.g., least-used ones).
        
        Args:
            router_logits: Original routing logits [batch_size, seq_len, num_experts]
            excluded_experts: List of expert indices to exclude
            
        Returns:
            Modified expert indices [batch_size, seq_len, top_k]
        """
        # Mask out excluded experts
        modified_logits = router_logits.clone()
        for expert_idx in excluded_experts:
            modified_logits[:, :, expert_idx] = float('-inf')
        
        # Select top-k from remaining experts
        expert_indices = torch.topk(modified_logits, k=self.top_k, dim=-1).indices
        
        return expert_indices
    
    def load_aware_assignment(self, router_logits: torch.Tensor, 
                             temperature: float = 2.0) -> torch.Tensor:
        """
        Balance assignments based on current expert load with temperature scaling.
        
        Args:
            router_logits: Original routing logits [batch_size, seq_len, num_experts]
            temperature: Temperature for softening the distribution
            
        Returns:
            Modified expert indices [batch_size, seq_len, top_k]
        """
        # Scale logits by temperature
        scaled_logits = router_logits / temperature
        
        # Get current load across the batch
        routing_probs = torch.softmax(scaled_logits, dim=-1)
        batch_load = routing_probs.sum(dim=(0, 1))
        
        # Compute load-aware penalties (higher load = higher penalty)
        avg_load = batch_load.mean()
        load_penalty = (batch_load - avg_load) / (avg_load + 1e-10)
        
        # Apply penalty
        adjusted_logits = scaled_logits - load_penalty.unsqueeze(0).unsqueeze(0)
        
        # Select top-k
        expert_indices = torch.topk(adjusted_logits, k=self.top_k, dim=-1).indices
        
        return expert_indices
    
    def apply_strategy(self, router_logits: torch.Tensor, 
                      strategy: AssignmentStrategy,
                      **kwargs) -> torch.Tensor:
        """
        Apply a specific assignment strategy.
        
        Args:
            router_logits: Original routing logits
            strategy: Strategy to apply
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Modified expert indices
        """
        if strategy == AssignmentStrategy.BALANCED:
            return self.balanced_assignment(router_logits)
        elif strategy == AssignmentStrategy.ROUND_ROBIN:
            return self.round_robin_assignment(router_logits)
        elif strategy == AssignmentStrategy.RANDOM:
            return self.random_assignment(router_logits)
        elif strategy == AssignmentStrategy.TOP_K_FILTERED:
            excluded = kwargs.get('excluded_experts', [])
            return self.top_k_filtered_assignment(router_logits, excluded)
        elif strategy == AssignmentStrategy.LOAD_AWARE:
            temp = kwargs.get('temperature', 2.0)
            return self.load_aware_assignment(router_logits, temp)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class MoELayerWrapper(nn.Module):
    """Wrapper for MoE layers that allows assignment modification."""
    
    def __init__(self, original_layer, modifier: ExpertAssignmentModifier, 
                 strategy: Optional[AssignmentStrategy] = None):
        """
        Initialize wrapper.
        
        Args:
            original_layer: Original MoE layer
            modifier: Assignment modifier instance
            strategy: Assignment strategy (None = use original routing)
        """
        super().__init__()
        self.original_layer = original_layer
        self.modifier = modifier
        self.strategy = strategy
        self.enabled = False
        
    def enable_modification(self, strategy: AssignmentStrategy):
        """Enable assignment modification with specified strategy."""
        self.strategy = strategy
        self.enabled = True
        
    def disable_modification(self):
        """Disable assignment modification (use original routing)."""
        self.enabled = False
        
    def forward(self, *args, **kwargs):
        """Forward pass with optional assignment modification."""
        if not self.enabled or self.strategy is None:
            # Use original routing
            return self.original_layer(*args, **kwargs)
        
        # Intercept and modify routing
        # This implementation depends on the specific MoE layer structure
        # Here's a generic approach:
        
        # Get hidden states
        hidden_states = args[0]
        
        # Get router logits
        if hasattr(self.original_layer, 'router'):
            router_logits = self.original_layer.router(hidden_states)
        else:
            # Fallback to original if router not found
            return self.original_layer(*args, **kwargs)
        
        # Modify expert assignments
        modified_indices = self.modifier.apply_strategy(router_logits, self.strategy)
        
        # Forward with modified assignments
        # This would need to be adapted based on the specific MoE implementation
        # For now, return original output (this is a placeholder)
        return self.original_layer(*args, **kwargs)


def wrap_moe_layers(model, modifier: ExpertAssignmentModifier) -> Dict[str, MoELayerWrapper]:
    """
    Wrap all MoE layers in the model for assignment modification.
    
    Args:
        model: The MoE model
        modifier: Assignment modifier instance
        
    Returns:
        Dictionary mapping layer names to wrappers
    """
    wrappers = {}
    
    for name, module in model.named_modules():
        if 'moe' in name.lower() and hasattr(module, 'router'):
            wrapper = MoELayerWrapper(module, modifier)
            wrappers[name] = wrapper
            
            # Replace the layer in the model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, wrapper)
    
    return wrappers


if __name__ == "__main__":
    # Example usage
    num_experts = 8
    top_k = 2
    
    modifier = ExpertAssignmentModifier(num_experts, top_k)
    
    # Simulate router logits
    router_logits = torch.randn(2, 10, num_experts)  # [batch, seq_len, num_experts]
    
    # Test different strategies
    for strategy in AssignmentStrategy:
        print(f"\nTesting {strategy.value} strategy:")
        indices = modifier.apply_strategy(router_logits, strategy)
        print(f"Shape: {indices.shape}")
        print(f"Sample indices: {indices[0, :3]}")
