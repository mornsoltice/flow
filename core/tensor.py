import numpy as np


class Tensor:
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}

        # Unique ID for each tensor
        self.id = id if id else np.random.randint(0, 100000)

        if creators:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def __repr__(self):
        return f"Tensor({self.data}, autograd={self.autograd})"

    def tambah(self, other):
        return Tensor(
            self.data + other.data,
            autograd=True,
            creators=[self, other],
            creation_op="tambah",
        )

    def kali(self, other):
        return Tensor(
            self.data * other.data,
            autograd=True,
            creators=[self, other],
            creation_op="kali",
        )

    def matmul(self, other):
        return Tensor(
            self.data.dot(other.data),
            autograd=True,
            creators=[self, other],
            creation_op="matmul",
        )

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is None:
                grad_origin = self

            if self.creation_op == "tambah":
                self.creators[0].backward(grad, grad_origin=self)
                self.creators[1].backward(grad, grad_origin=self)

            elif self.creation_op == "kali":
                new_grad = self.grad * grad
                self.creators[0].backward(new_grad, grad_origin=self)
                self.creators[1].backward(new_grad, grad_origin=self)

            elif self.creation_op == "matmul":
                new_grad_A = grad.matmul(self.creators[1].data.T)
                new_grad_B = self.creators[0].data.T.matmul(grad)
                self.creators[0].backward(Tensor(new_grad_A), grad_origin=self)
                self.creators[1].backward(Tensor(new_grad_B), grad_origin=self)

            if self.id in grad_origin.children:
                self.children[self.id] -= 1

            if self.children[self.id] == 0:
                self.grad = grad
                for c in self.creators:
                    c.backward(self.grad, grad_origin=self)
