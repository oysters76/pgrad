from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch
from pgrad.nn import MLP 
from pgrad.engine import Value

class MNIST:
    def __init__(self, train_subset=1000, test_subset=200, lr=0.01, ep=5):
        self.mnist, self.X, self.y = self.load_dataset() 
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split(self.X, self.y, train_subset, test_subset) 
        self.lr = lr 
        self.model = MLP(784, [30, 10])
        self.epoch_size = ep 
 
    def load_dataset(self):
        print("fetching mnist dataset...")
        mnist: Bunch = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')  # type: ignore[assignment]
        X, y = mnist.data, mnist.target.astype(int)
        print("fetched everything :)")
        return mnist, X, y 

    def train_test_split(self, x, y, train_subset, test_subset):
        x = x/255.0 
        x_train, y_train = x[:train_subset], y[:train_subset] 
        offset = train_subset + test_subset 
        x_test, y_test = x[train_subset:offset], y[train_subset:offset] 
        return x_train, y_train, x_test, y_test

    def one_hot(self, label, n_classes=10):
        return [1.0 if i == label else 0.0 for i in range(n_classes)] 

    def fit(self):
        for epoch in range(self.epoch_size):
            total_loss = 0.0 
            correct = 0 

            for i in range(len(self.X_train)): 
                inputs = [Value(float(x)) for x in self.X_train[i]] 
                result = self.model(inputs)
                target = self.one_hot(self.y_train[i])
                scores: list[Value] = result if isinstance(result, list) else [result]
                loss = sum(((s-t)**2 for s,t in zip(scores,target)), Value(0)) / len(scores)
                total_loss += loss.data

                pred = max(range(10), key=lambda k:scores[k].data)
                if pred == self.y_train[i]:
                    correct += 1

                for p in self.model.parameters():
                    p.grad = 0.0 

                loss.backward()

                for p in self.model.parameters(): 
                    p.data -= self.lr * p.grad 

                if i % 20 == 0:
                    print(f" epoch {epoch}, sample {i}, loss={loss.data:.4f}")

            print(f"Epoch {epoch}: avg_loss={total_loss/len(self.X_train):.4f},acc={correct/len(self.X_train):.3f}")

    def eval(self):
        correct = 0 
        for i in range(len(self.X_test)):
            inputs = [Value(float(x)) for x in self.X_test[i]] 
            result = self.model(inputs)
            scores: list[Value] = result if isinstance(result, list) else [result]
            pred = max(range(10), key=lambda k: scores[k].data)
            if pred == self.y_test[i]:
                correct += 1 
        print(f"Test Accuracy: {correct/len(self.X_test):.3f}") 
    

if __name__ == '__main__':
    clf = MNIST(train_subset=1000, test_subset=200, lr=0.0001, ep=3)
    clf.fit()
    clf.eval()
    