import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        fractions = []

        curr = x
        for layer in model.children():
            curr = layer(curr)

            if isinstance(layer, nn.ReLU):
                fractions.append(round((curr == 0).all(dim=0).float().mean().item(), 4))

        return fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        func_map: dict = {
            "use_leaky_relu": lambda: any(f > 0.5 for f in dead_fractions),
            "reinitialize": lambda: dead_fractions and dead_fractions[0] > 0.3,
            "reduce_learning_rate": lambda: dead_fractions and all(dead_fractions[i - 1] < dead_fractions[i] for i in range(1, len(dead_fractions))) \
                                    and dead_fractions[-1] > 0.1,
            "healthy": lambda: max(dead_fractions) < 0.1
        }

        for val, func in func_map.items():
            if func():
                return val

        return "healthy"
