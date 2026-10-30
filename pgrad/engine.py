from __future__ import annotations 
from collections.abc import Callable 

class Value:
    type Scalar = int | float 
    _scalar_types = (int,float)
    type ValueLike = Value | Value.Scalar 

    def __init__(self, data:Value.Scalar, children:tuple[Value,...]=(), label:str=''):
        self.data:Value.Scalar = data 
        self._children:tuple[Value,...] = children
        self._label:str = label
        self.grad:Value.Scalar = 0
        self._backward:Callable[[],None] = lambda: None # headless function that returns null

    def __wrap_value_with_other(self, other:ValueLike) -> Value:
        if not isinstance(other, Value):
            return Value(other)
        return other

    def __repr__(self) -> str:
        return f"Value(data={self.data},grad={self.grad})"

    def __add__(self, other: ValueLike) -> Value:
        other = self.__wrap_value_with_other(other)  
        out = Value(self.data + other.data, (self, other), '+') 

        def _backward():
            # for addition, we distribute the gradients to its children
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = _backward

        return out 

    def __mul__(self, other: ValueLike) -> Value:
        other = self.__wrap_value_with_other(other)
        out = Value(self.data * other.data, (self, other), '*') 

        def _backward():
            # we swap the data of self,other and mult in the input gradient flowing from upstream
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out

    def __pow__(self, other: Value.Scalar) -> Value:
        assert isinstance(other, Value._scalar_types), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'^{other}')

        def _backward():
            self.grad += (other * (self.data** (other-1))) * out.grad 
        out._backward = _backward

        return out

    def __radd__(self,other: Value.Scalar) -> Value:
        return self.__add__(other)

    def __rmul__(self,other: Value.Scalar) -> Value:
        return self.__mul__(other)

    def __neg__(self) -> Value:
        return self.__mul__(-1)

    def relu(self) -> Value:
        # if you dont pass a trailing comma into (self,) it treats self as a not a tuple value
        out = Value(max(0, self.data), (self,), 'relu') 

        def _backward():
            # if the data (activation) is greater than 0, pass the gradient downstream
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward 
        
        return out 

    def __sub__(self, other: ValueLike) -> Value:
        ot = self.__wrap_value_with_other(other) 
        return self.__add__(ot.__neg__()) 

    def __rsub__(self, other: Value.Scalar) -> Value:
        ot = self.__wrap_value_with_other(other)
        return ot.__add__(self.__neg__()) 

    def __truediv__(self, other: ValueLike) -> Value:
        ot = self.__wrap_value_with_other(other)
        return self.__mul__(ot.__pow__(-1)) 

    def __rtruediv__(self, other: Value.Scalar) -> Value:
        return Value(other).__mul__(self.__pow__(-1))

    def __topo_recursive(self, node: Value) -> list[Value]:
        topo = [] 
        visited = set() 
        def build_topo(v):
            if v not in visited: 
                visited.add(v) 
                for child in v._children:
                    build_topo(child) 
                topo.append(v) 
        build_topo(node)
        return topo

    def __topo_iterative(self, node: Value) -> list[Value]:
        topo = [] # this list will hold nodes in dependency order (leaves first, root last)
        visited = set()  # keep track of visited nodes
        stck = [(node, False)] # root node, I haven't explored all my childrens yet
        while len(stck) != 0:
            n, is_processed = stck.pop() 
            if is_processed:
                # safe to add me now, all my children are already in the topo list, 'post-order' moment
                topo.append(n) 
                continue
            if n in visited:
                continue 
            # mark this node as seen so we don't visit it again 
            visited.add(n) 
            # push myself again so add me to 'topo' list after all my children are done
            stck.append((n, True))

            # push all my children onto the stack, reversed() so they come off the stack in the same left-to-right order as recursive version 
            for child in reversed(n._children):
                if child not in visited: 
                    stck.append((child, False))
        return topo 

    def backward(self) -> None:
        topo = self.__topo_iterative(self)
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1 
        for n in reversed(topo):
            n._backward() 

