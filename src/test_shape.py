#!/usr/bin/env python3
"""Quick test to verify the model shape."""
import sys
sys.path.insert(0, '.')
import torch
from model import EEG2Gait, build_model

# Fake adjacency matrix (59x59)
A_init = torch.eye(59)
model = build_model(A_init=A_init)
print('Model created successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

print('\nArchitecture (paper components):')
for name, mod in model.named_children():
    print(f'  {name}: {mod.__class__.__name__}')

x = torch.randn(4, 1, 59, 100)
print(f'\nInput shape:  {x.shape}')

model.eval()
with torch.no_grad():
    y = model(x)
print(f'Output shape: {y.shape}')
assert y.shape == (4, 6), f'Shape mismatch! Got {y.shape}'
print('\n✅ PASSED — LTL → GCM → HGP → GSL → FFN → GTL → Output')
