from mpi4py import MPI

from PyTrilinos2.PyTrilinos2 import ROL
from PyTrilinos2.PyTrilinos2 import Teuchos
from PyTrilinos2.PyTrilinos2 import Tpetra
from PyTrilinos2.getTpetraTypeName import *
import numpy as np
from numpy import linalg as LA

tpetraVectorType = getTypeName('Vector')


class npVector(ROL.Vector_double_t):
    def __init__(self, dimension=1, default_value=0., values=None):
        if values is None:
            self.values = default_value*np.ones((dimension,))
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
        #return npVector(values=np.copy(self.values))
        return npVector(dimension=len(self.values))
    def axpy(self, scale_factor, x):
        ax = x.clone()
        ax.plus(x)
        ax.scale(scale_factor)
        self.plus(ax)
    def dimension(self):
        return len(self.values)
    def setScalar(self, new_value):
        self.values[:] = new_value
    def __getitem__(self, index):
        return self.values[index]
    def __setitem__(self, index, val):
        self.values[index] = val
    # To implement: applyUnary, applyBinary, reduce, randomize * 3


class tVector(ROL.Vector_double_t):
    def __init__(self, dimension=1, default_value=0., map=None, comm=None):
        if map is None:
            if comm is None:
                comm = Teuchos.getTeuchosComm(MPI.COMM_WORLD)
            map = getTypeName('Map')(dimension, 0, comm)
        self.tvector = tpetraVectorType(map, False)
        self.tvector.putScalar(default_value)
        super().__init__()
    def plus(self, b):
        self.tvector.update(1., b.tvector, 1.)
    def scale(self, scale_factor):
        self.tvector.scale(scale_factor)
    def dot(self, b):
        return self.tvector.dot(b.tvector)
    def norm(self):
        return self.tvector.norm2()
    def clone(self):
        return tVector(map=self.tvector.getMap())
    def axpy(self, scale_factor, x):
        self.tvector.update(scale_factor, x.tvector, 1.)
    def dimension(self):
        return self.tvector.getMap().getGlobalNumElements()
    def setScalar(self, new_value):
        self.tvector.putScalar(new_value)
    def __getitem__(self, index):
        map = self.tvector.getMap()
        if map.isNodeGlobalElement(index):
            local_index = map.getLocalElement(index)
            view = self.tvector.getLocalViewHost()
            return view[local_index]
    def __setitem__(self, index, val):
        map = self.tvector.getMap()
        if map.isNodeGlobalElement(index):
            local_index = map.getLocalElement(index)
            view = self.tvector.getLocalViewHost()
            view[local_index] = val
            self.tvector.setLocalViewHost(view)
    # To implement: applyUnary, applyBinary, reduce, randomize * 3

vector_type = tVector

obj = ROL.Objective_double_t()
c = ROL.Constraint_double_t()
a = vector_type(10, 1.)
b = vector_type(10, 1.)
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
print(b[0])
b[0]=-1.
print(b[0])