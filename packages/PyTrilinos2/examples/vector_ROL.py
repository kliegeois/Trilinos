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
        if isinstance( index, int ):
            map = self.tvector.getMap()
            if map.isNodeGlobalElement(index):
                local_index = map.getLocalElement(index)
                view = self.tvector.getLocalViewHost()
                return view[local_index]
        if isinstance( index, slice ):
            map = self.tvector.getMap()
            view = self.tvector.getLocalViewHost()
            global_indices = range(*index.indices(self.dimension()))
            local_indices = np.empty(np.size(global_indices), dtype=int)
            for i in range(0, len(global_indices)):
                if map.isNodeGlobalElement(global_indices[i]):
                    local_indices[i] = map.getLocalElement(global_indices[i])
                else:
                    local_indices[i] = 0
            return view[local_indices]
    def __setitem__(self, index, val):
        if isinstance( index, int ):
            map = self.tvector.getMap()
            if map.isNodeGlobalElement(index):
                local_index = map.getLocalElement(index)
                view = self.tvector.getLocalViewHost()
                view[local_index] = val
                self.tvector.setLocalViewHost(view)
        if isinstance( index, slice ):
            map = self.tvector.getMap()
            view = self.tvector.getLocalViewHost()
            global_indices = range(*index.indices(self.dimension()))
            local_indices = np.empty(np.size(global_indices), dtype=int)
            for i in range(0, len(global_indices)):
                if map.isNodeGlobalElement(global_indices[i]):
                    local_indices[i] = map.getLocalElement(global_indices[i])
                else:
                    local_indices[i] = 0
            view[local_indices] = val
            self.tvector.setLocalViewHost(view)
    # To implement: applyUnary, applyBinary, reduce, randomize * 3

class norm2Obj(ROL.Objective_double_t):
    def __init__(self, target=None):
        self.target = target
        super().__init__()
    def setTarget(self, target):
        self.target = target
    def value(self, x, tol):
        if self.target is None:
            return x.norm()
        tmp = x.clone()
        tmp.plus(x)
        tmp.axpy(-1, self.target)
        return tmp.norm()

# Matrix from rol/example/quadratic/example_01.cpp
class matrix(ROL.LinearOperator_double_t):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()
    def apply(self, Hv, v, tol):
        for i in range(0, self.dim):
            Hv[i] = 2.*v[i]
            if i > 0:
                Hv[i] -= v[i-1]
            if i < self.dim - 1:
                Hv[i] -= v[i+1]


vector_type = tVector

obj = norm2Obj()
op = matrix(10)
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
b[0:2]=[-1., 3.]
print(b[0:3])

print(obj.value(a, 1e-8))

op.apply(b,a,1e-8)
print(b[0:3])
g = vector_type(10, 1.)

params = ROL.getParametersFromXmlFile("input.xml")

print(params)

#obj = ROL.QuadraticObjective_double_t(op, g)