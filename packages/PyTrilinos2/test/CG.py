import unittest
from mpi4py import MPI
import numpy as np
from PyTrilinos2.PyTrilinos2 import Teuchos
from PyTrilinos2.PyTrilinos2 import Tpetra
from PyTrilinos2.getTpetraTypeName import getTypeName

def CG(A, x, b, max_iter=20, tol=1e-8):
    r = type(b)(b, Teuchos.DataAccess.Copy)
    A.apply(x,r,Teuchos.ETransp.NO_TRANS,alpha=-1,beta=1)

    p = type(r)(r, Teuchos.DataAccess.Copy)
    q = type(r)(r, Teuchos.DataAccess.Copy)

    gamma = r.norm2()
    print('gamma_init = ' +str(gamma))
    if gamma < tol:
        return
    for j in range(0, max_iter):
        A.apply(p,q)
        c = q.dot(p)
        alpha = gamma**2 / c
        x.update(alpha, p, 1)
        r.update(-alpha, q, 1)
        gamma_next = r.norm2()
        print('gamma_'+str(j)+' = ' +str(gamma_next))
        beta = gamma_next**2/gamma**2
        gamma = gamma_next
        if gamma < tol:
            return
        p.update(1, r, beta)

class TestCG(unittest.TestCase):
    def test_all(self):
        comm = Teuchos.getTeuchosComm(MPI.COMM_WORLD)

        mapType = getTypeName('Map')
        graphType = getTypeName('CrsGraph')
        matrixType = getTypeName('CrsMatrix')
        vectorType = getTypeName('Vector')
        multivectorType = getTypeName('MultiVector')

        mapT=mapType(14,0,comm)
        print(mapT)
        print('mapT.getMinLocalIndex() = '+str(mapT.getMinLocalIndex()))
        print('mapT.getMaxLocalIndex() = '+str(mapT.getMaxLocalIndex()))
        print('mapT.getMinGlobalIndex() = '+str(mapT.getMinGlobalIndex()))
        print('mapT.getMaxGlobalIndex() = '+str(mapT.getMaxGlobalIndex()))
        print('Tpetra.getDefaultComm().getSize() = '+str(Tpetra.getDefaultComm().getSize()))
        mv=multivectorType(mapT, 3, True)
        mv.replaceLocalValue(0,0,1.23)
        mv.replaceLocalValue(0,1,1.23)
        #mv.randomize(0,-2)
        v0=mv.getVector(0)
        v1=mv.getVector(1)
        print(mv.description())
        print(v0.description())
        print(v0.norm2())
        print(v0.dot(v1))

        graph = graphType(mapT,3)
        for i in range(mapT.getMinLocalIndex(), mapT.getMaxLocalIndex()+1):
            global_i = mapT.getGlobalElement(i)
            graph.insertGlobalIndices(global_i, [global_i])
        graph.fillComplete()

        A = matrixType(graph)

        for i in range(mapT.getMinLocalIndex(), mapT.getMaxLocalIndex()+1):
            global_i = mapT.getGlobalElement(i)
            A.replaceGlobalValues(global_i, [global_i], [2.])

        A.fillComplete()

        print(A.getGlobalNumEntries())
        print(A.description())
        print(A.getFrobeniusNorm())

        print(v0.norm2())
        print(v1.norm2())
        A.apply(v0,v1)
        print(v1.norm2())


        x=vectorType(mapT, True)
        b=vectorType(mapT, False)

        b.randomize(0,-2)

        print('Norm of x before CG = '+str(x.norm2()))
        print('Norm of b = '+str(b.norm2()))
        CG(A, x, b, max_iter=5)
        print('Norm of x after CG = '+str(x.norm2()))

        self.assertAlmostEqual(2*x.norm2(), b.norm2(), delta=1e-5)

        local_x = x.getLocalViewHost()
        local_b = b.getLocalViewHost()

        self.assertAlmostEqual(np.linalg.norm(2*local_x-local_b), 0., delta=1e-5)


if __name__ == '__main__':
    unittest.main()