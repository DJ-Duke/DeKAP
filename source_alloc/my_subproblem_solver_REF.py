import numpy as np
from scipy.optimize import linprog
import time
def my_subproblem_solver(y_cur, mproblem_dict):
    # y_cur 转化为 bool 
    T_scaled = mproblem_dict["T_scaled"].copy()
    eta_1_Loss = mproblem_dict["eta_1_Loss"]
    eta_3 = mproblem_dict["eta_3"]
    N = mproblem_dict["N"]
    L = mproblem_dict["L"]
    y_cur = y_cur.astype(bool)
    for i in range(N):
        for l in range(L):
            T_scaled[i,i,l] = 0 # 这一步也可以放在函数外面。
            index = i*L + l
            if not y_cur[index]: T_scaled[i,:,l] = float("inf")
    
    T_min_Rx = np.min(T_scaled, axis = 0) # T_scaled (N, N, L) --> (N,L)
    T_min_Rx_1 = np.expand_dims(T_min_Rx, axis = 0)
    T_min_Rx_2 = np.expand_dims(T_min_Rx, axis = 1)
    T_min_Rx_i = np.argmin(T_scaled, axis = 0)

    e_ijl_opt = np.zeros((N*(N-1)*L), dtype=bool)
    phi_hijl_opt = np.zeros((N*(N-1)**2*L), dtype=bool)
    vphi_hijl_opt = np.zeros((N*(N-1)**2*L), dtype=bool)

    eta_3_T_min_ij_l = eta_3 * (np.tile(T_min_Rx_1, (N,1,1)) + np.tile(T_min_Rx_2, (1,N,1)))
    e_ij_opt = np.argmin(eta_1_Loss + eta_3_T_min_ij_l, axis = 2)
    # 用e_ij_opt 每个元素的值更新 e_ij_l_opt
    for i in range(N):
        for j in range(N):
            if j == i: continue
            l = e_ij_opt[i,j]
            j_adjusted = j if j < i else j-1
            index = i*(N-1)*L + j_adjusted*L + l
            e_ijl_opt[index] = 1
            
            temp = index * (N-1)
            h = T_min_Rx_i[i,l]
            if h != i: 
                h_adjusted = h if h < i else h-1
                index_ = temp + h_adjusted
                phi_hijl_opt[index_] = 1
            h = T_min_Rx_i[j,l]
            if h != j: 
                h_adjusted = h if h < j else h-1
                index_ = temp + h_adjusted
                vphi_hijl_opt[index_] = 1
    x_opt = np.concatenate([e_ijl_opt, phi_hijl_opt, vphi_hijl_opt])
    return x_opt

def compute_dual_solution(A, b, c, x_star, tol=1e-6):
    """
    计算线性规划的对偶问题最优解 y*，给定原问题的最优解 x*。

    参数：
    A      : numpy.ndarray  -> 形状 (m, n) 的约束矩阵
    b      : numpy.ndarray  -> 形状 (m,) 的约束向量
    c      : numpy.ndarray  -> 形状 (n,) 的目标系数
    x_star : numpy.ndarray  -> 形状 (n,) 的原问题最优解
    tol    : float          -> 判断活跃约束的数值精度

    返回：
    y_star : numpy.ndarray  -> 形状 (m,) 的对偶问题最优解
    """
    m, n = A.shape

    # 1. 找出活跃约束（tight constraints）
    active_constraints = np.isclose(A @ x_star, b, atol=tol)

    # 2. 找出非零 x* 的索引
    not_zero_x_condition = x_star > tol


    # 3. 组合两个条件
    A_active = A[active_constraints]  # 只保留活跃约束的行

    # 4. 构建完整的线性方程组 A_eq * y = c
    A_eq = A_active.T  # 需要确保这些列能解出 c
    A_eq = A_eq[not_zero_x_condition] # 筛选出非零的列
    c_eq = c[not_zero_x_condition]

    # # 5. 直接求解
    # if A_eq.shape[0] >= A_eq.shape[1]:
    #     y_star, _, _, _ = np.linalg.lstsq(A_eq, c_eq, rcond=None)  # 最小二乘求解
    # else:
    #     y_star = np.linalg.solve(A_eq, c_eq)
    y_star = np.linalg.pinv(A_eq) @ c_eq

    # 6. 组装对偶解（非活跃约束的 y 设为 0）
    y_dual = np.zeros(m)
    y_dual[active_constraints] = y_star[:sum(active_constraints)]

    # 7. 确保 y* >= 0
    if np.any(y_dual < -tol):
        print("警告: 求得的 y* 存在负值，可能需要调整活跃约束。")

    return y_dual

def solving_dual_solution(A, b, c, x_star, tol=1e-6):
    '''
    min cx, s.t., Ax <=b, x>=0
    '''

    # 首先，变形成为 Min cx, s.t., Ax >= b, x>=0, 其标准对偶形式 max by, A^T y <= c, y>=0


    # 首先，确定紧约束
    u_star = np.zeros(len(b))

    active_constraints = np.isclose(A @ x_star, b, atol=tol)

    zero_x_condition = x_star < tol
    not_zero_x_condition = np.logical_not(zero_x_condition)

    b_hat = b[active_constraints]
    A_hat = A[active_constraints, :]
    A_hat_x0 = A_hat[:, zero_x_condition]
    A_hat_x1 = A_hat[:, not_zero_x_condition]
    c_x0 = c[zero_x_condition]
    c_x1 = c[not_zero_x_condition]
    bounds = [(0, None) for _ in range(len(b_hat))]

    result_1 = linprog(-b_hat, A_ub=A_hat.T, b_ub=c, A_eq=A_hat_x1.T, b_eq=c_x1, bounds=bounds, method="highs")
    if result_1.success:
        u_star[active_constraints] = result_1.x
    else:
        raise ValueError("求解对偶问题失败")

    return u_star

def solving_dual_solution_scipy(A, b, c, tol = 1e-6):
    bounds_full = [(0, None) for _ in range(len(b))]
    start_time = time.time()
    result = linprog(-b, A_ub=A.T, b_ub=c, bounds=bounds_full, method="highs")
    end_time = time.time()
    return result.x, end_time - start_time

def solving_dual_solution_my(A, b, c, x_star, tol = 1e-6):
    # 首先，确定紧约束
    active_constraints = np.isclose(A @ x_star, b, atol=tol)

    zero_x_condition = x_star < tol
    not_zero_x_condition = np.logical_not(zero_x_condition)

    b_hat = b[active_constraints]
    A_hat = A[active_constraints, :]
    A_hat_x0 = A_hat[:, zero_x_condition]
    A_hat_x1 = A_hat[:, not_zero_x_condition]
    c_x0 = c[zero_x_condition]
    c_x1 = c[not_zero_x_condition]
    bounds = [(0, None) for _ in range(len(b_hat))]

    start_time = time.time()
    result = linprog(-b_hat, A_ub=A_hat_x0.T, b_ub=c_x0, A_eq=A_hat_x1.T, b_eq=c_x1, bounds=bounds, method="highs")
    end_time = time.time()
    time_1 = end_time - start_time
    return result.x, time_1