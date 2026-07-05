from typing import List
from heapq import heappop, heappush
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        curr_items: list[str] = list(corpus)
        list_of_merges = []

        for ii in range(num_merges):
            print(ii, curr_items)

            freq = {}
            to_replace = defaultdict(set)
            curr_heap = []

            for i in range(1, len(curr_items)):
                curr_pair = (curr_items[i - 1], curr_items[i])
                
                freq[curr_pair] = freq.get(curr_pair, 0) + 1
                to_replace[curr_pair].add(i - 1)

                heappush(curr_heap, (-freq[curr_pair], curr_pair))

            _, best_pair = heappop(curr_heap)
            list_of_merges.append(list(best_pair))
                
            i = 0
            new_curr_items = []

            stringified = best_pair[0] + best_pair[1]
            while i < len(curr_items):
                if i in to_replace[best_pair]:
                    new_curr_items.append(stringified)
                    i += 2
                else:
                    new_curr_items.append(curr_items[i])
                    i += 1

            curr_items = new_curr_items

        return list_of_merges


