from ..diabtools import *
import os
import numpy as np
import pytest

class TestSymMat:
    
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
        testdata = np.linspace(0,1,4)[:,np.newaxis]
        X,Y,Z = np.meshgrid(testdata, testdata, testdata)
        data = np.hstack((
            X.flatten()[:,np.newaxis],
            Y.flatten()[:,np.newaxis],
            Z.flatten()[:,np.newaxis],
            ))
        Wx = M(data)
        assert Wx.shape == (len(testdata)**Nd, Ns, Ns)
        assert all([np.allclose(Wx[i,:,:], Wx[i,:,:].T) for i in range(len(testdata))])

    def test_common_shift(self):
        W = SymPolyMat.zero(3,2)
        for i in range(3):
            for j in range(i+1):
                n = i+j
                W[i,j][(0,0)] = n
                W[i,j][(1,0)] = n+1
                W[i,j][(0,1)] = n+2
                W[i,j][(1,1)] = n+3
        W.set_common_x0(np.array([1.,1.]))
        assert all([np.all(p.x0 == np.array([1.,1.])) for p in W])
        assert np.all(W([1., 1.]) == np.array([[0,1,2],[1,2,3],[2,3,4]]))
        
    def test_separate_shifts(self):
        W = SymPolyMat.zero(3,2)
        for i in range(3):
            for j in range(i+1):
                n = i+j
                W[i,j][(0,0)] = n
                W[i,j][(1,0)] = n+1
                W[i,j][(0,1)] = n+2
                W[i,j][(1,1)] = n+3
                W[i,j].x0 = [0.1*i, 0.2*j]
        allx0 = W.get_all_x0()
        assert all([
            np.all(W[0,0].x0 == np.array([0.0, 0.0])),
            np.all(W[1,0].x0 == np.array([0.1, 0.0])),
            np.all(W[2,0].x0 == np.array([0.2, 0.0])),
            np.all(W[1,1].x0 == np.array([0.1, 0.2])),
            np.all(W[2,1].x0 == np.array([0.2, 0.2])),
            np.all(W[2,2].x0 == np.array([0.2, 0.4])),
            ])
        assert all([W[i,j](W[i,j].x0) == i+j for i in range(3) for j in range(i+1)])

    def test_file_write_read(self):
        Wout = SymPolyMat.zero(3,3)
        for i in range(3):
            for j in range(i+1):
                Wout[i,j].grow_degree(i+j+1, fill=2*i-j)

        filename = "test.spm"
        Wout.write_to_file(filename)
        Win = SymPolyMat.read_from_file(filename)

        os.remove(filename)
        for wo, wi in zip(Wout, Win):
            assert wo == wi
            assert np.all(wo.x0 == wi.x0)
        
    @pytest.mark.skip(reason="No way of reading from text.")
    def test_write_txt(self):
        Wout = SymPolyMat.zero(3,3)
        for i in range(3):
            for j in range(i+1):
                Wout[i,j].grow_degree(i+j+1, fill=2*i-j)

        filename = "test.txt"
        Wout.write_to_txt(filename)
        # Win = SymPolyMat.read_from_txt(filename)
        os.remove(filename)

    def test_coeffs_and_keys(self):
        W = SymPolyMat.zero(3,3)
        for i in range(3):
            for j in range(i+1):
                W[i,j].grow_degree(1, fill = i + j)
        coeffs_array, keys_list = W.coeffs_and_keys()
        
        exp_coeffs = []
        exp_keys = []
        for i in range(3):
            for j in range(i+1):
                # Degree 0 monomial
                exp_coeffs += [0]     
                exp_keys += [(i,j,(0,0,0))]

                # Degree 1 monomials (there are 3)
                exp_coeffs += [i + j for _ in range(3)]
                exp_keys += [(i,j,(0,0,1)), (i,j,(0,1,0)), (i,j,(1,0,0))]

        exp_coeffs = np.array(exp_coeffs)

        assert np.all(coeffs_array == exp_coeffs)
        assert all([k == ek for k, ek in zip(keys_list,exp_keys)])


