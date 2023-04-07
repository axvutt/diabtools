from ..diabtools import *
import numpy as np

class TestSymMat:
    testdata = np.linspace(0,1,4)[:,np.newaxis]
    
    def test_Symmetry(self):
        Ns = 2
        Nd = 3
        M = SymPolyMat(Nd,Ns)
        M[0,0] = NdPoly.zero(Nd)
        M[1,0] = NdPoly.one(Nd)
        M[1,1] = NdPoly({(1,2,3): 4})
        assert M[1,0] == M[0,1]

    def test_Empty(self):
        Ns = 3
        Nd = 4
        M = SymPolyMat(Ns, Nd)
        assert all([len(m.powers()) == 0 for m in M])
        assert all([m.Nd == Nd for m in M])

    def test_Zero(self):
        Ns = 3
        Nd = 4
        M = SymPolyMat.zero(Ns, Nd)
        assert all([list(m.powers()) == [m.zeroPower] for m in M])
        assert all([list(m.coeffs()) == [0] for m in M])
        assert all([m.Nd == Nd for m in M])

    def test_ZeroLike(self):
        Ns = 3
        Nd = 2
        A = SymPolyMat(Ns,Nd)
        A[0,0] = NdPoly({(0,1): 2, (3,4): 5})
        A[1,0] = NdPoly({(5,4): 3, (2,1): 0})
        A[1,1] = NdPoly({(1,1): 1, (2,2): 2})
        A[2,0] = NdPoly({(3,3): 3, (4,4): 4})
        A[2,1] = NdPoly({(3,1): 4, (4,1): 3})
        A[2,2] = NdPoly({(0,4): 2, (6,6): 6})
        B = SymPolyMat.zero_like(A)
        assert all([list(a.powers()) == list(b.powers()) for a,b in zip(A,B)])
        assert all([list(b.coeffs()) == [0 for _ in range(len(b))] for b in B])

    def test_Eye(self):
        Ns = 3
        Nd = 2
        I = SymPolyMat.eye(Ns,Nd)
        assert all([list(i.powers()) == [i.zeroPower] for i in I])
        assert all([list(I[i,j].coeffs()) == [0] for i in range(Ns) for j in range(i)])
        assert all([list(I[i,i].coeffs()) == [1] for i in range(Ns)])

    def test_EyeLike(self):
        Ns = 3
        Nd = 2
        A = SymPolyMat(Ns,Nd)
        A[0,0] = NdPoly({(0,1): 2, (3,4): 5})
        A[1,0] = NdPoly({(5,4): 3, (2,1): 0})
        A[1,1] = NdPoly({(1,1): 1, (2,2): 2})
        A[2,0] = NdPoly({(3,3): 3, (4,4): 4})
        A[2,1] = NdPoly({(3,1): 4, (4,1): 3})
        A[2,2] = NdPoly({(0,4): 2, (6,6): 6})
        B = SymPolyMat.eye_like(A)
        assert all([list(B[i,j].powers()) == list(A[i,j].powers()) for i in range(Ns) for j in range(i)])
        assert all([a_power in B[i,i] for i in range(Ns) for a_power in list(A[i,i].powers())])
        assert all([B[i,i][B[i,i].zeroPower] == 1 for i in range(Ns)])
     
    def test_Call(self):
        Ns = 5
        Nd = 3
        M = SymPolyMat.zero(Ns,Nd)
        P = NdPoly({(1,2,3): 0.1, (3,0,0): 3.14})
        M[0,0] = P
        M[1,1] = NdPoly({(1,0,0): 1, (0,0,1): 2})
        M[2,2] = P
        M[1,0] = NdPoly({(0,0,0): 1})
        M[0,2] = NdPoly({(0,0,0): 0.1})
        M[3,0] = NdPoly({(0,0,1): 6.66})
        X,Y,Z = np.meshgrid(self.testdata, self.testdata, self.testdata)
        data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            Z.flatten()[:,np.newaxis],
            ))
        Wx = M(data)
        assert Wx.shape == (len(self.testdata)**Nd, Ns, Ns)
        assert all([np.allclose(Wx[i,:,:], Wx[i,:,:].T) for i in range(len(self.testdata))])

    def test_shift(self):
       pass 
