import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        diagnosis = []
        with torch.no_grad():

            curr = x
            for layer in model.children():
                curr = layer(curr)

                if isinstance(layer, nn.Linear):
                    diagnosis.append({
                        'mean': round(curr.mean().item(), 4),
                        'std': round(curr.std().item(), 4),
                        'dead_fraction': round((curr <= 0).all(dim=0).float().mean().item(), 4)
                    })

        return diagnosis

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        loss = nn.MSELoss()

        y_hat = model(x)
        output = loss(y_hat, y)
        output.backward()

        diagnosis = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                g = layer.weight.grad

                diagnosis.append({
                    'mean': round(g.mean().item(), 4),
                    'std': round(g.std().item(), 4),
                    'norm': round(g.norm().item(), 4)
                })

        return diagnosis



    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        n = len(activation_stats)

        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5: return 'vanishing_gradients'

        for i in range(n):
            act_st = activation_stats[i]
            grad_st = gradient_stats[i]

            if act_st['dead_fraction'] > 0.5: return 'dead_neurons'
            if grad_st['norm'] > 1000: return 'exploding_gradients'

            if act_st['std'] < 0.1: return 'vanishing_gradients'
            if act_st['std'] > 10: return 'exploding_gradients'

        return 'healthy'

