"""
Expert imbalance loss for fine-tuning MoE models.
Encourages more balanced expert utilization during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ExpertImbalanceLoss(nn.Module):
    """Loss function to penalize expert load imbalance."""
    
    def __init__(self, 
                 num_experts: int,
                 loss_weight: float = 0.01,
                 loss_type: str = "gini"):
        """
        Initialize imbalance loss.
        
        Args:
            num_experts: Total number of experts
            loss_weight: Weight for the imbalance loss term
            loss_type: Type of imbalance metric ('gini', 'variance', 'cv', 'entropy')
        """
        super().__init__()
        self.num_experts = num_experts
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        
    def compute_gini_coefficient(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute Gini coefficient as a measure of imbalance.
        
        Args:
            expert_probs: Expert usage probabilities [num_experts]
            
        Returns:
            Gini coefficient (0 = perfect balance, 1 = complete imbalance)
        """
        # Sort probabilities
        sorted_probs, _ = torch.sort(expert_probs)
        
        # Compute Gini coefficient
        n = self.num_experts
        cumsum = torch.cumsum(sorted_probs, dim=0)
        gini = (n + 1 - 2 * torch.sum(cumsum) / (cumsum[-1] + 1e-10)) / n
        
        return gini
    
    def compute_variance_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute variance from uniform distribution.
        
        Args:
            expert_probs: Expert usage probabilities [num_experts]
            
        Returns:
            Variance loss
        """
        uniform_prob = 1.0 / self.num_experts
        variance = torch.mean((expert_probs - uniform_prob) ** 2)
        return variance
    
    def compute_cv_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute coefficient of variation.
        
        Args:
            expert_probs: Expert usage probabilities [num_experts]
            
        Returns:
            CV loss
        """
        mean_prob = torch.mean(expert_probs)
        std_prob = torch.std(expert_probs)
        cv = std_prob / (mean_prob + 1e-10)
        return cv
    
    def compute_entropy_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute negative entropy (maximize entropy for balance).
        
        Args:
            expert_probs: Expert usage probabilities [num_experts]
            
        Returns:
            Negative entropy loss
        """
        entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10))
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=expert_probs.dtype, 
                                             device=expert_probs.device))
        # Normalize and invert (we want to maximize entropy)
        normalized_entropy = entropy / max_entropy
        return 1.0 - normalized_entropy
    
    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute imbalance loss.
        
        Args:
            router_logits: Routing logits [batch_size, seq_len, num_experts]
            
        Returns:
            Imbalance loss value
        """
        # Convert logits to probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Aggregate probabilities across batch and sequence
        expert_probs = routing_probs.mean(dim=(0, 1))  # [num_experts]
        
        # Compute the appropriate loss
        if self.loss_type == "gini":
            loss = self.compute_gini_coefficient(expert_probs)
        elif self.loss_type == "variance":
            loss = self.compute_variance_loss(expert_probs)
        elif self.loss_type == "cv":
            loss = self.compute_cv_loss(expert_probs)
        elif self.loss_type == "entropy":
            loss = self.compute_entropy_loss(expert_probs)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.loss_weight * loss


class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss similar to the auxiliary loss used in Switch Transformer.
    Encourages uniform expert usage.
    """
    
    def __init__(self, num_experts: int, loss_weight: float = 0.01):
        """
        Initialize load balancing loss.
        
        Args:
            num_experts: Total number of experts
            loss_weight: Weight for the load balancing loss
        """
        super().__init__()
        self.num_experts = num_experts
        self.loss_weight = loss_weight
        
    def forward(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            router_logits: Routing logits [batch_size, seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            
        Returns:
            Load balancing loss
        """
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Compute fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1, 2))  # [num_experts]
        fraction_per_expert = tokens_per_expert / (batch_size * seq_len * expert_indices.shape[-1])
        
        # Compute average routing probability to each expert
        avg_routing_prob = routing_probs.mean(dim=(0, 1))  # [num_experts]
        
        # Load balancing loss: product of fractions and probabilities scaled by num_experts
        loss = self.num_experts * torch.sum(fraction_per_expert * avg_routing_prob)
        
        return self.loss_weight * loss


class CombinedMoELoss(nn.Module):
    """
    Combined loss for MoE training including:
    - Language modeling loss
    - Expert imbalance loss
    - Load balancing loss
    """
    
    def __init__(self,
                 num_experts: int,
                 imbalance_weight: float = 0.01,
                 load_balance_weight: float = 0.01,
                 imbalance_type: str = "gini"):
        """
        Initialize combined loss.
        
        Args:
            num_experts: Number of experts
            imbalance_weight: Weight for imbalance loss
            load_balance_weight: Weight for load balancing loss
            imbalance_type: Type of imbalance metric
        """
        super().__init__()
        self.imbalance_loss = ExpertImbalanceLoss(
            num_experts, imbalance_weight, imbalance_type
        )
        self.load_balance_loss = LoadBalancingLoss(
            num_experts, load_balance_weight
        )
        
    def forward(self, 
                lm_loss: torch.Tensor,
                router_logits_list: List[torch.Tensor],
                expert_indices_list: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.
        
        Args:
            lm_loss: Language modeling loss
            router_logits_list: List of routing logits from each MoE layer
            expert_indices_list: List of expert indices from each layer (optional)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        total_imbalance_loss = 0.0
        total_load_balance_loss = 0.0
        
        for i, router_logits in enumerate(router_logits_list):
            # Imbalance loss
            total_imbalance_loss += self.imbalance_loss(router_logits)
            
            # Load balancing loss (if expert indices provided)
            if expert_indices_list is not None and i < len(expert_indices_list):
                total_load_balance_loss += self.load_balance_loss(
                    router_logits, expert_indices_list[i]
                )
        
        # Average across layers
        num_layers = len(router_logits_list)
        avg_imbalance_loss = total_imbalance_loss / num_layers
        avg_load_balance_loss = total_load_balance_loss / num_layers if expert_indices_list else 0.0
        
        # Total loss
        total_loss = lm_loss + avg_imbalance_loss + avg_load_balance_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'imbalance_loss': avg_imbalance_loss.item(),
            'load_balance_loss': avg_load_balance_loss.item() if expert_indices_list else 0.0
        }
        
        return total_loss, loss_dict


class MoEModelWithImbalanceLoss(nn.Module):
    """
    Wrapper for MoE model that adds imbalance loss during training.
    """
    
    def __init__(self, 
                 model,
                 num_experts: int,
                 imbalance_weight: float = 0.01,
                 load_balance_weight: float = 0.01):
        """
        Initialize wrapper.
        
        Args:
            model: Base MoE model
            num_experts: Number of experts per layer
            imbalance_weight: Weight for imbalance loss
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__()
        self.model = model
        self.combined_loss = CombinedMoELoss(
            num_experts, imbalance_weight, load_balance_weight
        )
        self.router_logits_cache = []
        
    def cache_router_logits(self):
        """Register hooks to cache router logits during forward pass."""
        self.router_logits_cache = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'router'):
                router_output = module.router(input[0])
                if isinstance(router_output, tuple):
                    self.router_logits_cache.append(router_output[0])
                else:
                    self.router_logits_cache.append(router_output)
        
        # Register hooks on MoE layers
        for name, module in self.model.named_modules():
            if 'moe' in name.lower() and hasattr(module, 'router'):
                module.register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with imbalance loss.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for language modeling
            
        Returns:
            If labels provided: (total_loss, loss_dict)
            Otherwise: model outputs
        """
        self.router_logits_cache = []
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if labels is not None and len(self.router_logits_cache) > 0:
            # Compute combined loss
            lm_loss = outputs.loss
            total_loss, loss_dict = self.combined_loss(
                lm_loss, self.router_logits_cache
            )
            
            return total_loss, loss_dict
        
        return outputs


if __name__ == "__main__":
    # Example usage
    num_experts = 8
    batch_size = 4
    seq_len = 128
    
    # Simulate router logits
    router_logits = torch.randn(batch_size, seq_len, num_experts)
    expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, 2))
    
    # Test different loss types
    for loss_type in ["gini", "variance", "cv", "entropy"]:
        imbalance_loss = ExpertImbalanceLoss(num_experts, loss_weight=0.01, loss_type=loss_type)
        loss = imbalance_loss(router_logits)
        print(f"{loss_type.capitalize()} loss: {loss.item():.4f}")
    
    # Test load balancing loss
    lb_loss = LoadBalancingLoss(num_experts, loss_weight=0.01)
    loss = lb_loss(router_logits, expert_indices)
    print(f"\nLoad balancing loss: {loss.item():.4f}")
