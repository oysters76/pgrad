# pgrad

A tiny autograd engine and neural network library built from scratch in Python. Implements reverse-mode automatic differentiation over a dynamically built computation graph, similar to PyTorch's autograd.

## Features

- **Value**: scalar-valued autograd engine with support for `+`, `*`, `**`, `/`, `-`, `relu`, and automatic gradient computation via backpropagation
- **Neural network modules**: `Neuron`, `Layer`, and `MLP` built on top of the autograd engine

## Usage

```python
from pgrad.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
d.backward()

print(a.grad)  # dd/da
print(b.grad)  # dd/db
```

```python
from pgrad.nn import MLP

model = MLP(3, [4, 4, 1])
x = [Value(1.0), Value(2.0), Value(3.0)]
output = model(x)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Testing

```bash
python tests/test_engine.py
```

Tests compare pgrad's forward and backward passes against PyTorch to verify correctness.

## Dev tools

```bash
pip install ruff pyright pre-commit
pre-commit install
```

Pre-commit hooks run ruff (format + lint) and pyright (typecheck) on every commit.
