import numpy as np

# def max(A, B):
#     return A if A > B else B

# def min(A, B):
#     return A if A < B else B

def forwardrecursion(n, F, Fi, Fj, dmin):
    tmin=np.zeros(n)
    lpathtoin=np.zeros(n)
    for i in range(n):
        tmin[i] = 0
        lpathtoin[i] = 0
        k = 0
        while Fj[i + 1] - Fj[i] - k > 0:
            j = Fj[i] + k
            fig = Fi[j]
            tmin[i] = max(tmin[i], tmin[fig])
            lpathtoin[i] = max(lpathtoin[i], lpathtoin[fig])
            k += 1
        tmin[i] += dmin[i]
        lpathtoin[i] += 1
    return tmin, lpathtoin

def backwardrecursion(n, Ft, Fti, Ftj, dmin, T):
    tmax=np.zeros(n)
    lpathtoout=np.zeros(n)
    for i in range(n - 1, -1, -1):
        tmax[i] = T
        lpathtoout[i] = 0
        k = 0
        while Ftj[i + 1] - Ftj[i] - k > 0:
            j = Ftj[i] + k
            fog = Fti[j]
            tmax[i] = min(tmax[i], tmax[fog] - dmin[fog])
            lpathtoout[i] = max(lpathtoout[i], lpathtoout[fog] + 1)
            k += 1
    return tmax, lpathtoout

def mexFunction_slck(F, Ft, dmin, T):
    n = F.shape[0]
    
    # Convert CSR format (Compressed Sparse Row) to index arrays
    Fi = F.indices
    Fj = F.indptr
    Fti = Ft.indices
    Ftj = Ft.indptr
    
    tmin, lpathtoin = forwardrecursion(n, F.toarray(), Fi, Fj, dmin)
    tmax, lpathtoout = backwardrecursion(n, Ft.toarray(), Fti, Ftj, dmin, T)

    return tmin, tmax, lpathtoin, lpathtoout

# # Example usage
# if __name__ == "__main__":
#     # Example values for F, Ft, dmin, T, tmin, tmax, lpathtoin, lpathtoout
#     F = np.array([[0, 1], [1, 2], [2, 0]])  # Replace with your matrix data
#     Ft = np.array([[0, 1], [1, 2], [2, 0]])  # Replace with your matrix data
#     dmin = np.array([1.0, 2.0, 3.0])  # Example distances
#     T = np.array([5.0])  # Example maximum time
#     tmin = np.zeros(3)  # Output array for tmin
#     tmax = np.zeros(3)  # Output array for tmax
#     lpathtoin = np.zeros(3)  # Output array for lpathtoin
#     lpathtoout = np.zeros(3)  # Output array for lpathtoout

#     # Convert F and Ft to CSR format
#     from scipy.sparse import csr_matrix
#     F_csr = csr_matrix(F)
#     Ft_csr = csr_matrix(Ft)

#     # Call the function
#     mexFunction_slck(F_csr, Ft_csr, dmin, T, tmin, tmax, lpathtoin, lpathtoout)

#     print("tmin:", tmin)
#     print("tmax:", tmax)
#     print("lpathtoin:", lpathtoin)
#     print("lpathtoout:", lpathtoout)
