from PyTrilinos2.PyTrilinos2 import ROL
import numpy as np
from numpy import linalg as LA


class myVector(ROL.Vector_double_t):
    def __init__(self, length=1, default_value=0., values=None):
        if values is None:
            self.values = default_value*np.ones((length,))
        else:
            self.values = values
        super().__init__()
    def plus(self, b):
        self.values += b.values
    def scale(self, scale_factor):
        self.values *= scale_factor
    def dot(self, b):
        return np.dot(self.values, b.values)
    def norm(self):
        return LA.norm(self.values)
    def clone(self):
        return myVector(values=np.copy(self.values))
    def axpy(self, scale_factor, x):
        ax = x.clone()
        ax.scale(scale_factor)
        self.plus(ax)
    def setScalar(self, new_value):
        self.values[:] = new_value
    # To implement: applyUnary, applyBinary, reduce, randomize * 3

obj = ROL.Objective_double_t()
c = ROL.Constraint_double_t()
a = myVector(10, 1.)
b = myVector(10, 1.)
a.scale(2.)
print(a.norm())
a.zero()
print(a.norm())

a.axpy(1., b)
print(a.norm())
print(a.dot(b))

b.setScalar(2.)
a = b
print(a.apply(b))
print(b.norm())
