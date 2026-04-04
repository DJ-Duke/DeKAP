# Cython implementation of the per-fixed-y subproblem (continuous x).
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.uint8_t BTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def my_subproblem_solver_cy(np.ndarray[BTYPE_t, ndim=1] y_cur,
                           dict problem_dict):
    cdef:
        np.ndarray[DTYPE_t, ndim=3] T_scaled_ = problem_dict["T_scaled"]
        np.ndarray[DTYPE_t, ndim=3] eta_1_Loss_arr = problem_dict["eta_1_Loss"]
        DTYPE_t eta_3 = problem_dict["eta_3"]
        int N = problem_dict["N"]
        int L = problem_dict["L"]

        np.ndarray[DTYPE_t, ndim=3] T_scaled = np.empty((N, N, L), dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=2] T_min_Rx
        np.ndarray[ITYPE_t, ndim=2] T_min_Rx_i
        np.ndarray[BTYPE_t, ndim=1] e_ijl_opt, phi_hijl_opt, vphi_hijl_opt
        np.ndarray[DTYPE_t, ndim=3] eta_3_T_min_ij_l
        np.ndarray[ITYPE_t, ndim=2] e_ij_opt

        int i, j, l, h, index, temp, index_
        Py_ssize_t idx_i, idx_j, idx_l
        DTYPE_t min_val, current_val
        ITYPE_t min_idx

    for i in range(N):
        for j in range(N):
            for l in range(L):
                T_scaled[i,j,l] = T_scaled_[i,j,l]

    y_cur_bool = y_cur.astype(bool)

    for i in range(N):
        for l in range(L):
            T_scaled[i,i,l] = 0
            index = i*L + l
            if not y_cur_bool[index]:
                for j in range(N):
                    T_scaled[i,j,l] = INFINITY

    T_min_Rx = np.zeros((N, L), dtype=np.float64)
    T_min_Rx_i = np.zeros((N, L), dtype=np.int64)

    for j in range(N):
        for l in range(L):
            min_val = INFINITY
            min_idx = 0
            for i in range(N):
                current_val = T_scaled[i,j,l]
                if current_val < min_val:
                    min_val = current_val
                    min_idx = i
            T_min_Rx[j,l] = min_val
            T_min_Rx_i[j,l] = min_idx

    cdef np.ndarray[DTYPE_t, ndim=2] cumu_T_min_Rx = np.zeros((N, L), dtype=np.float64)
    for j in range(N):
        cumu_T_min_Rx[j,0] = T_min_Rx[j,0]
        for l in range(1, L):
            cumu_T_min_Rx[j,l] = cumu_T_min_Rx[j,l-1] + T_min_Rx[j,l]

    e_ijl_opt = np.zeros(N*(N-1)*L, dtype=np.uint8)
    phi_hijl_opt = np.zeros(N*(N-1)**2*L, dtype=np.uint8)
    vphi_hijl_opt = np.zeros(N*(N-1)**2*L, dtype=np.uint8)

    eta_3_T_min_ij_l = np.zeros((N, N, L), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            for l in range(L):
                eta_3_T_min_ij_l[i,j,l] = eta_3 * (cumu_T_min_Rx[i,l] + cumu_T_min_Rx[j,l])

    e_ij_opt = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            min_val = INFINITY
            min_idx = 0
            for l in range(L):
                current_val = eta_1_Loss_arr[i,j,l] + eta_3_T_min_ij_l[i,j,l]
                if current_val < min_val:
                    min_val = current_val
                    min_idx = l
            e_ij_opt[i,j] = min_idx

    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            l = e_ij_opt[i,j]
            j_adjusted = j if j < i else j-1
            index = i*(N-1)*L + j_adjusted*L + l
            e_ijl_opt[index] = 1

            for l_1 in range(0, l+1):
                temp = (index - l + l_1) * (N-1)
                h = T_min_Rx_i[i,l_1]
                if h != i:
                    h_adjusted = h if h < i else h-1
                    index_ = temp + h_adjusted
                    phi_hijl_opt[index_] = 1

                h = T_min_Rx_i[j,l_1]
                if h != j:
                    h_adjusted = h if h < j else h-1
                    index_ = temp + h_adjusted
                    vphi_hijl_opt[index_] = 1

    return np.concatenate([e_ijl_opt, phi_hijl_opt, vphi_hijl_opt])
