import sys
import torch

def prepare_sequence(batch_data, word_ix, device):
    return [torch.tensor([word_ix[w] for w in data], dtype=torch.long, device=device) for data in batch_data]

def padding_function(batch_data, word_ix, device):
    batch_data = prepare_sequence(batch_data, word_ix, device)
    batch_data = sorted(batch_data, key=lambda data:data.size()[-1], reverse=True)
    X_lengths = [x.size()[-1] for x in batch_data]
    max_length = max(X_lengths)
    
    x_padding = torch.zeros(len(batch_data), max_length, dtype=torch.long, device=device)
    for index, data in enumerate(batch_data):
        x_padding[index, :X_lengths[index]] = data
    return x_padding, X_lengths