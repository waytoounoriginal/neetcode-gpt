import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.
        result = []

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        for i in range(new_chars):
            if len(context.T) > context_length:
                context = context[:, -context_length:]

            logits = model(context)
            preds = torch.softmax(logits[:, -1, :], dim=-1)

            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            next_char = torch.multinomial(preds, 1, generator=generator)
            # MORE OF YOUR CODE (arbitrary number of lines)

            print(next_char)
            context = torch.cat((context, next_char), dim=1)
            result.append(int_to_char[next_char.item()])

        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
        return ''.join(result)
