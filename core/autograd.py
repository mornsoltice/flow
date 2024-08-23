class Node:
    def __init__(self, data, parents=[]):
        self.data = data
        self.parents = parents
        self.grad = 0

    def backward(self, grad=1):
        self.grad += grad
        for parent in self.parents:
            parent.backward(grad)


class Tensor(Node):
    def tambah(self, other):
        out = Tensor(self.data + other.data, parents=[self, other])
        return out

    def kali(self, other):
        out = Tensor(self.data * other.data, parents=[self, other])
        return out
