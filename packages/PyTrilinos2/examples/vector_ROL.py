from PyTrilinos2.PyTrilinos2 import ROL
import numpy as np
from numpy import linalg as LA
print(ROL)

class myVector(ROL.Vector_double_t):
    def __init__(self, length=1, default_value=0.):
        self.values = default_value*np.ones((length,))
        super().__init__()
    def plus(self, b):
        self.values += b.values
    def scale(self, scale_factor):
        self.values *= scale_factor
    def dot(self, b):
        return np.dot(self.values, b.values)
    def norm(self):
        return LA.norm(self.values)
    def clone(self, b):
        self.values = np.copy(b.values)



a = myVector(10, 1.)
b = myVector(10, 1.)
a.scale(2.)
print(a.norm())
a.zero()
print(a.norm())

a.plus(b)
print(a.norm())
print(a.dot(b))