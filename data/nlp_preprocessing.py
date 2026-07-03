import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        words = []
        words.extend([w for s in positive + negative for w in s.split()])
        words.sort()


        vocab = {}
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab) + 1

        tensors = []
        for s in positive:
            tensors.append(
                torch.tensor(
                    [vocab[w] for w in s.split()]
                )
            )

        for s in negative:
            tensors.append(
                torch.tensor(
                    [vocab[w] for w in s.split()]
                )
            )

        paddings = nn.utils.rnn.pad_sequence(tensors, padding_value=0, batch_first=True)
        return paddings
        

