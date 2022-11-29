from mpi4py import MPI

from PyTrilinos2.PyTrilinos2 import ROL
from PyTrilinos2.PyTrilinos2 import Teuchos
from PyTrilinos2.getTpetraTypeName import *
import numpy as np
from numpy import linalg as LA


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
    def __init__(self, dimension=1, default_value=0., map=None, comm=None, scalar_type=getDefaultScalarType(), local_ordinal_type=getDefaultLocalOrdinalType(), global_ordinal_type=getDefaultGlobalOrdinalType(), node_type=getDefaultNodeType()):
        self.scalar_type = scalar_type
        self.local_ordinal_type = local_ordinal_type
        self.global_ordinal_type = global_ordinal_type
        self.node_type = node_type
        self.mapType = getTypeName('Map', local_ordinal_type=self.local_ordinal_type, global_ordinal_type=self.global_ordinal_type, node_type=self.node_type)
        self.vectorType = getTypeName('Vector', scalar_type=self.scalar_type, local_ordinal_type=self.local_ordinal_type, global_ordinal_type=self.global_ordinal_type, node_type=self.node_type)
        if map is None:
            if comm is None:
                comm = Teuchos.getTeuchosComm(MPI.COMM_WORLD)
            map = self.mapType(dimension, 0, comm)
        self.tvector = self.vectorType(map, False)
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
        return tVector(map=self.tvector.getMap(), scalar_type=self.scalar_type, local_ordinal_type=self.local_ordinal_type, global_ordinal_type=self.global_ordinal_type, node_type=self.node_type)
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
                self.tvector.replaceGlobalValue(index, val)
        if isinstance( index, slice ):
            map = self.tvector.getMap()
            global_indices = range(*index.indices(self.dimension()))
            for i in range(0, len(global_indices)):
                if map.isNodeGlobalElement(global_indices[i]):
                    if len(val) > 1:
                        self.tvector.replaceGlobalValue(global_indices[i], val[i])
                    else:
                        self.tvector.replaceGlobalValue(global_indices[i], val)
    # To implement: applyUnary, applyBinary, reduce, randomize * 3

class norm2Obj(ROL.Objective_double_t):
    def __init__(self, H, g, c=0):
        self.H = H
        self.g = g
        self.c = c
        super().__init__()
    def value(self, x, tol):
        tmp = x.clone()
        self.H.apply(tmp, x, tol)
        tmp.scale(0.5)
        tmp.plus(self.g)
        return x.apply(tmp) + self.c
    def gradient(self, g, x, tol):
        self.H.apply(g, x, tol)
        g.plus(self.g)
    def hessVec(self, hv, v, x, tol):
        self.H.apply(hv, v, tol)
    def invHessVec(self, hv, v, x, tol):
        self.H.applyInverse(hv, v, tol)

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


op = matrix(10)
g = vector_type(10, 1.)
x = vector_type(10, 0.)
obj = norm2Obj(op, g)
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

problem = ROL.Problem_double_t(obj, x)
#solver = ROL.Solver_double_t(problem, params)

#print(params)

#obj = ROL.QuadraticObjective_double_t(op, g)