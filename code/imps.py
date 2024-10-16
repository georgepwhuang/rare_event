
#%%
from numpy.core.fromnumeric import argmax
import scipy.linalg as LA
from scipy.sparse.base import isspmatrix
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import scipy.stats as stats

import numpy as np

import scipy as sp
import scipy.sparse as sps
# from . import HMM



#%%

'''
The iMPS class
'''

class iMPS(object):
    def __init__(self,smatrix):

        #The construction is from kraus operators
        self. _smatrix = smatrix
        self._alp_size = len(smatrix)
        self._dim, _ = smatrix[0].shape
        self._ld_eigval = None
        self._Vr = None
        self._Vl = None
        self._E = None
        self._shape = (self.alp_size,self.dim)
        self.schmidt_coefficients = None

    @classmethod
    def from_tmatrix(cls, tmatrix):
        smatrix = np.sqrt(tmatrix)

        return cls(smatrix)

    @property
    def smatrix(self):
        return self. _smatrix

    @smatrix.setter
    def smatrix(self, smatrix):
        self.__init__(smatrix)

    @property
    def dim(self):
        return self._dim

    @property
    def alp_size(self):
        return self._alp_size

    @property
    def shape(self):
        return self._shape

    @property
    def Vl(self):
        if self._Vl is None:
            self.ld_eig()
        return self._Vl
    
    @property
    def Vr(self):
        if self._Vr is None:
            self.ld_eig()
        return self._Vr

    @property
    def ld_eigval(self):
        if self._ld_eigval is None:
            self.ld_eig()
        return self._ld_eigval

    @property
    def E(self):
        if self._E is None:
            tran_dim = self._dim**2
            self._E = np.zeros((tran_dim, tran_dim), dtype=complex)
            for i in range(self.alp_size):
                self._E += LA.kron(self.smatrix[i], np.conjugate(self.smatrix[i]))
        return self._E

    '''
    Returns the leading eigenvalue, leading left eigenvector and leading right eigenvector
    '''
    def ld_eig(self):
        if self._ld_eigval is None:
            lamb,V = LA.eig(self.E)
            lamb = np.abs(lamb)
            i = np.argmax(lamb)

            # set the leading eigenvalue
            self._ld_eigval = lamb[i]

            # set the leading right eigenvector
            self._Vr = np.reshape(V[:,i],(self.dim,self.dim))
            # self._Vr /= trace(self.Vr[0,0])

            # set the leading left eigenvector
            lamb, V = LA.eig(np.transpose(self.E))
            lamb = np.abs(lamb)
            i = np.argmax(lamb)
            self._Vl = np.reshape(V[:, i],(self.dim, self.dim))
            # Nomalize Vr and Vl b
            self._Vl = np.transpose(self._Vl)

            # Set a restriction np.trace(Vl*Vr) = 1.0 by dividing the normalizing factor

            nm_factor = np.trace(self.Vl.dot(self.Vr))
            self._Vl /= np.sqrt(nm_factor)
            self._Vr /= np.sqrt(nm_factor)



        return self._ld_eigval,self._Vl,self._Vr



    '''
    Evaluate the left canonical form of the iMPS
    '''
    def lft_cform(self):
        if self._ld_eigval is None:
            self.ld_eig()
        Wl = LA.sqrtm(self.Vl)
        inv_Wl = LA.inv(Wl)
        lft_cform = np.array([Wl.dot(matrix).dot(inv_Wl) for matrix in self. _smatrix])
        lft_cform /= np.sqrt(self._ld_eigval)
        
        return iMPS(lft_cform)

    '''
    Evaluate the right canonical form of the iMPS
    '''
    def rgt_cform(self):
        if self._ld_eigval is None:
            self.ld_eig()
        Wr = LA.sqrtm(self._Vr)
        inv_Wr = LA.inv(Wr)
        rgt_cform = np.array([inv_Wr.dot(matrix).dot(Wr)
                           for matrix in self. _smatrix])
        rgt_cform /= np.sqrt(self._ld_eigval)

        return iMPS(rgt_cform)
        # print(y)


    '''
     Return the canonical form 
     Return the gamma, and the Schmit coefficients
    '''
    def cform(self):

        nm = np.trace(self.Vl*self.Vr)
        wl = LA.sqrtm(self.Vl/nm)
        wr = LA.sqrtm(self.Vr)

        U,S, Vh = LA.svd(wl.dot(wr))

        def wrap(A):
            return Vh.dot(LA.inv(wr)).dot(A).dot(LA.inv(wl)).dot(U)/np.sqrt(self.ld_eigval)
        
        smatrix = []
        for mm in self.smatrix:
            smatrix.append(wrap(mm))

        gamma = iMPS(np.array(smatrix))
        return gamma, S

    
    '''
    Return the quantum model
    1. The Kraus operators
    2. The states
    '''
    def qmodel(self):
        if self._ld_eigval is None:
            self.ld_eig()

        self._Vl /= self._Vl[0, 0]
        self._Vr /= np.trace(self._Vr)

        kop = self.lft_cform()
        states = LA.sqrtm(self.Vl)

        return kop, states
    
    def get_schmidt_coefficients(self):
        "get the schmidt coefficients in descending order."


        if self.schmidt_coefficients is None:
            Wl = LA.sqrtm(self.Vl)
            rho = Wl.dot(self.Vr).dot(Wl.T.conj())
            lam = LA.eigvalsh(rho)
            lam = np.abs(np.real(lam))

            lam /= sum(lam)
            # Ignore numerical negative
            for i in range(len(lam)):
                if np.abs(lam[i]) < 1e-10:
                    lam[i] = 0
            lam = np.sqrt(lam)
            self.schmidt_coefficients = np.sort(lam)[::-1]
        return self.schmidt_coefficients



    def cq(self, base = 2):
        '''
        Generate the cq of the quantum models, base =2
        '''

        lam = self.get_schmidt_coefficients()
        probability = np.power(lam,2)
        ans = stats.entropy(probability,base = base)

        return ans

    '''
    Generate the probability distribution of a given sequence.
    '''
    def log_prob_seq(self,seq,base = 2,state= None, past=None):

        # initialize the state
        if state is None:
            if past is None:
                rho = self.Vr/np.trace(self.Vr)
            else:
                v1 = self.en_past(past)
                v1 = np.reshape(v1,(len(v1),1))
                rho = LA.kron(v1,v1.T.conj())
        elif isinstance(state,int):
            rho = np.zeros((self.dim,self.dim))
            rho[state,state] = 1
        else:
            rho = state

        Vl = self.Vl/np.trace(self.Vl.dot(rho))

        log_p = 0.

        for x in seq:
            rho = self.smatrix[x].dot(rho).dot(self.smatrix[x].conj().T)
            p = np.real(np.trace(Vl.dot(rho)))
            if abs(p) < 1e-10:
                return -np.inf
            log_p += np.log2(p)

            rho /= p
        log_p /= np.log2(base)
        return log_p

        # pass
        
    '''
    
    Sample a sequence of given length
    '''
    def sample_seq(self, length=100, state = None):

        # initialize the state
        if state is None:
            rho = self.Vr/np.trace(self.Vr)
        elif isinstance(state,int):
            rho = np.zeros((self.dim,self.dim))
            rho[state,state] = 1
        else:
            rho = state
        

        Vl = self.Vl/np.trace(self.Vl.dot(rho))

        seq = []
        for _ in range(length):
            p_out = np.zeros(self.alp_size)
            for j in range(self.alp_size):
                pc = np.trace(self.smatrix[j].dot(rho.dot(self.smatrix[j].conj().T)).dot(Vl))
                p_out[j] = np.real(pc)
            
            if abs(sum(p_out)-1.) > 1e-6:
                print(p_out)
                raise ValueError("It is not a distribution")
            x = np.random.choice(self.alp_size, p=p_out)
            seq.append(x)
            rho = self.smatrix[x].dot(rho.dot(self.smatrix[x].conj().T))
            rho /= np.trace(rho)
        

        return seq

    '''
    Find a unitary operator according to the site matrix.
    The first system is the memory system.
    The second system is the output system.
    '''
    def to_unitary(self,simulator = "default"):
        _smatrix = self.lft_cform().smatrix

        M,N = self.shape 

        Umatrix = np.zeros((M*N,M*N),dtype=complex)

        for x in range(M):
            for j in range(N):
                for k in range(N):
                    row_index = x + j*M
                    col_index = k*M
                    Umatrix[row_index,col_index] =  _smatrix[x,j,k]

        U,Pmatrix = LA.polar(Umatrix)



        ans_U = np.zeros((M*N,M*N),dtype=complex)
        if simulator == "default":
            ans_U = U
        if simulator == "qiskit":
            # Swap the order of the unitary operator
            for row_x in range(M):
                for row_j in range(N):
                    for column_x in range(M):
                        for column_j in range(N):
                            ans_U[row_x*N + row_j, column_x*N + column_j] = U[row_x + row_j*M, column_x + column_j*M]
                    

        return ans_U
    
    '''
    Evaluate the mixed gauge MPS
    '''
    def to_mixed_gauge(self):
        Wl = LA.sqrtm(self.Vl)
        Wr = LA.sqrtm(self.Vr)

        C = Wl.dot(Wr)

        Ac = [Wl.dot(mm).dot(Wr)/np.sqrt(self.ld_eigval) for mm in self.smatrix]

        return MixMPS([Ac, C])


    '''
    Truncate an imps to a given dimension.
    '''
    def trunc(self,tdim):


        lft_imps = self.lft_cform()
        lam, U = LA.eigh(lft_imps.Vr)
        abs_lam = abs(lam)
        idx = np.argsort(abs_lam)[::-1]

        lam = lam[idx]
        U = U[:,idx]

        tmatrix = []

        for A in lft_imps.smatrix:
            mm = U.conj().T.dot(A).dot(U)[0:tdim,0:tdim]
            tmatrix.append(mm)

        return iMPS(np.array(tmatrix))

    
    '''
    Truncate an imps to a given dimension using canonical form.
    '''
    def trunc_can(self,tdim):
        gamma,lam = self.cform()
        smatrix = np.array([np.diag(lam).dot(mm)[0:tdim,0:tdim] for mm in gamma.smatrix])

        return iMPS(smatrix)


    '''
    Compress the past into a pure quantum state.
    We assume the state is in its left canonical form.

    Parameters
    ----------

    past: array

    Return
    ---------
    v0: 1d array


    Note that this method works for process with finite Markov order.

    '''
    # def en_past(self, past):
    #     lam,V = LA.eig(self.smatrix[0])
    #     lam = np.abs(lam)
    #     i = lam.argmax()
    #     v0 = V[:,i]
    #     v0 /= LA.norm(v0)
    #     for x in past:
    #         v0 = self.smatrix[x].dot(v0)
    #         v0 /= LA.norm(v0)
    #     return v0

    def en_past(self, past,max_len = None):
        if max_len != None:
            past = past[-1*max_len:]
        rho = self.Vr
        
        for x in past:
            rho = self.smatrix[x].dot(rho).dot(self.smatrix[x].T.conj())
            # print("trace rho",rho.trace())
            rho /= np.real(rho.trace())
            # print("rho={},smatrix={}".format(rho,self.smatrix))
        lam,V = LA.eig(rho)
        i = np.argmax(np.abs(lam))
        v0 = V[:,i]
        return v0
    
    '''
    Generate the mps with sparse matrices
    '''

    def to_sparse_mps(self):
            smatrix = [csc_matrix(mm) for mm in self.smatrix]
            return iMPS(smatrix)




#%%

