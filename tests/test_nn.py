from pgrad.nn import MLP 
from pgrad.engine import Value

def test_finite_differences():
    eps = 0.001 
    inputs = [0.5, -0.3, 0.8]
    target = [1.0, 0.0] 
    model = MLP(3, [4, 2])

    xs = [Value(float(x)) for x in inputs] 
    result = model(xs) 
    scores = [result] if isinstance(result, Value) else result
    loss = sum(((s-t)**2 for s,t in zip(scores,target)),Value(0))  

    loss.backward() 

    for p in model.parameters():
        xs = [Value(float(x)) for x in inputs]
        p.data += eps 
        result = model(xs) 
        scores = [result] if isinstance(result, Value) else result
        new_loss = sum(((s-t)**2 for s,t in zip(scores,target)),Value(0))  
        finite_diff = (new_loss.data-loss.data)/(eps) 
        p.data -= eps 

        assert abs(finite_diff-p.grad) < 1e-2, f"mismatch: finite_diff={finite_diff}, grad={p.grad}"

    print("test_finite_differences passed!")