# %%
import numpy as np

# %%

'''

Convert a number arbitrary base sequence.

'''

def dec_to_seq(i, alphabet_size, seq_len):
    seq = []

    for _ in range(seq_len):
        seq.append(i % alphabet_size)
        i = i // alphabet_size
    # past = [int(symbol) for symbol in past]
    seq.reverse()

    return seq
# %%

'''

Convert a number arbitrary base sequence.

'''

def digits(i, base = 2, pad=2):
    seq = []

    for _ in range(pad):
        seq.append(i % base )
        i = i // base
    # past = [int(symbol) for symbol in past]
    seq.reverse()

    return seq

# %%

'''
Convert a digital seq into int
'''

def digits_to_int(seq,base=2):
    L = len(seq)
    ans = 0
    for i in range(L):
        ans += seq[i]*base**(L-i-1)
    return ans


# %%

# Several tool functions

def check_lft_cmp(imps):
    X = np.zeros((imps.dim,imps.dim),dtype=complex)
    for i in range(imps.alp_size):
        X += imps.smatrix[i].conj().T.dot(imps.smatrix[i])
    
    return X

def check_rgt_cmp(imps):
    mats = [imps.smatrix[i].dot(imps.smatrix[i].conj().T) for i in range(imps.alp_size)]

    return np.sum(mats,axis=0)


# %%
'''
Turn a unitary operator into a Kraus operators.
'''

def unitary_to_kraus(U,shape="default"):
    dim,_ = U.shape
    if shape == "default":
        shape = (2,np.int(np.sqrt(dim/2)))

    M,N = shape

    kraus_op = np.zeros((M,N,N),dtype=complex)

    for x in range(M):
        for j in range(N):
            for k in range(N):
                row_index = x + j*N
                col_index = k*N
                kraus_op[x,j,k,] =  U[row_index,col_index]
    
    return kraus_op




# %%
