import random 
from pgrad.engine import Value 
from abc import ABC, abstractmethod 

class Module(ABC):
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0 

    @abstractmethod
    def parameters(self) -> list[Value]:
        return [] 

class Neuron(Module):
    def __init__(self, input_size:int, nonlin:bool=True):
        self.w:list[Value] = [Value(random.uniform(-1,1)) for _ in range(input_size)] 
        self.b:Value = Value(0)
        self.nonlin:bool   = nonlin 

    # triggered when n = Neuron(..) n(params)
    def __call__(self, x:list[Value]|Value) -> Value:
        act:Value = self.b 
        x = [x] if isinstance(x, Value) else x 
        for wi,xi in zip(self.w, x):
            act = act + wi * xi 
        return act.relu() if self.nonlin else act 

    def parameters(self) -> list[Value]:
        return self.w + [self.b] 

    def __repr__(self) -> str:
        kind:str = 'ReLU' if self.nonlin else 'Linear' 
        size:int = len(self.w) 
        return f"{kind}Neuron({size})" 

class Layer(Module):
    def __init__(self, input_size:int, output_size:int, **kwargs):
        self.neurons:list[Neuron] = [Neuron(input_size, **kwargs) for _ in range(output_size)]

    def __call__(self, x:list[Value]|Value) -> list[Value]|Value:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out 

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin:int, nouts:list[int]):
        sz:list[int] = [nin] + nouts 
        self.layers:list[Layer] = [] 

        for i in range(len(nouts)):
            is_nonlinear:bool = i != len(nouts)-1  # if i isnt the last layer its non linear layer
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=is_nonlinear))

    def __call__(self, x:list[Value]|Value) -> list[Value]|Value:
        for layer in self.layers:
            x = layer(x) 
        return x 

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()] 

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
