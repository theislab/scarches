import numpy as np
import torch

unknown_key = 'unknown'
cell_types = ['a', 'b', 'c', 'unknown']
known_types = cell_types.copy()
if unknown_key in known_types:
    known_types.remove(unknown_key)

encoder = {k: v for k, v in zip(sorted(known_types), np.arange(len(known_types)))}
encoder[unknown_key] = -1
print(encoder)
print(cell_types)
all_labels = list()
for i in range(1):
    labels = np.zeros(1000)
    ad_cell_types = np.random.choice(cell_types, 1000, replace=True)
    for condition, label in encoder.items():
        labels[ad_cell_types == condition] = label
    all_labels.append(labels)

all_labels = np.stack(all_labels).T
all_labels = torch.tensor(all_labels, dtype=torch.long)
print(all_labels.size())

print(all_labels[1,:].size())

