from pgrad.engine import Value 

def softmax(logits: list[Value]) -> list[Value]:
    mx = max(v.data for v in logits) 
    sh = [(v-mx).exp() for v in logits] 
    total = sum(sh, Value(0)) 
    return [v/total for v in sh]

def cross_entropy(probs:list[Value], target:int) -> Value:
    return -probs[target].log()