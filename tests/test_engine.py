import torch
from pgrad.engine import Value 

def test_simple_case():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    g.backward() 

    test_cases = [
        (round(g.data,4), 24.7041, "g forward passed", "g forward failed"), 
        (round(a.grad,4), 138.8338, "a backward passed", "a backward failed"), 
        (round(b.grad,4), 645.5773, "b backward passed", "b backward failed"),
    ]

    for act,exp,passed,failed in test_cases:
        print(act,exp)
        assert act == exp, failed 
        print(passed)

    print("test-simple-case passed") 

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    assert ymg.data == ypt.data.item(),"sanity check failed for ymg == ypt"
    assert xpt.grad is not None 
    assert xmg.grad == xpt.grad.item(),"sanity check failed for xmg == xpt" 
    print("sanity-check-test passed")

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert apt.grad is not None 
    assert abs(amg.grad - apt.grad.item()) < tol
    assert bpt.grad is not None 
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print("test_more_ops passed")

if __name__ == '__main__':
    test_simple_case()
    test_sanity_check() 
    test_more_ops()