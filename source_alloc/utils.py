import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import milp, LinearConstraint, Bounds
from z_paper_evaluations.LP_test.subproblem_solver import my_subproblem_solver_cy
from z_paper_evaluations.LP_test.subproblem_solver_MK import my_subproblem_solver_cy_MK
from z_paper_evaluations.LP_test.my_subproblem_solver_REF import compute_dual_solution, my_subproblem_solver, solving_dual_solution
from scipy.io import loadmat
def assign_random_problem(seed=0, N=3, L=2):
    np.random.seed(seed)
    eta_1 = np.random.rand()
    eta_2 = np.random.rand()
    eta_3 = np.random.rand()
    F = np.random.rand(N, N)
    T = np.random.rand(N, N)
    T_scale = np.linspace(1, 5, L)
    Loss = np.random.rand(L)
    # sort loss with descending order
    Loss = np.sort(Loss)[::-1]
    CA = np.random.rand(L)
    # sort CA with ascending order
    CA = np.sort(CA)
    k = 0
    num_constraints = 2*N*(N-1)*L+2*N*(N-1)**2*L+2*N*(N-1) + 2*L
    num_variables = N*(N-1)*L+N*L+2*N*(N-1)**2*L
    problem = {
        "N": N,
        "L": L,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta_3": eta_3,
        "F": F,
        "T": T,
        "T_scale": T_scale,
        "Loss": Loss,
        "CA": CA,
        "k": k,
        "num_constraints": num_constraints,
        "num_variables": num_variables
    }
    print(f"num_ constraints: {num_constraints}, num_variables: {num_variables}")
    return problem

def half_randn(*args):
    return np.abs(np.random.randn(*args))

def problem_assign_formal_random_problem(seed=0, N=3, L=2):
    np.random.seed(seed)
    eta_1 = half_randn()  
    eta_2 = half_randn() 
    eta_3 = half_randn() 
    F = half_randn(N, N) 
    F = F / (np.sum(F) - np.sum(np.diag(F))) * N * (N-1)
    T = half_randn(N, N)
    T_scale = np.linspace(1, 5, L)
    loss_coeff_scale = np.linspace(1, 5, L)
    loss_coeff_factor_down = half_randn() 
    loss_coeff_factor_up = half_randn() 
    Loss = loss_coeff_factor_down * np.exp( - loss_coeff_scale * loss_coeff_factor_up )
    CA = (half_randn()) * T_scale
    k = 0
   
    # sort CA with ascending order
    num_constraints = 2*N*(N-1)*L+2*N*(N-1)**2*L+2*N*(N-1) + 2*L
    num_variables = N*(N-1)*L+N*L+2*N*(N-1)**2*L
    problem = {
        "seed": seed,
        "N": N,
        "L": L,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta_3": eta_3,
        "F": F,
        "T": T,
        "T_scale": T_scale,
        "Loss": Loss,
        "CA": CA,
        "k": k,
        "num_constraints": num_constraints,
        "num_variables": num_variables
    }
    print(f"[!!!] N = {N}, L = {L} problem, #constraints: {num_constraints}, #variables: {num_variables}")
    return problem

def formal_random_problem_with_training_loss(seed=0, N=3, L=2):
    np.random.seed(seed)
    eta_1 = 1 # loss weight
    eta_2 = 1 # storage cost weight
    eta_3 = 1 # transmission cost weight
    # eta_4 = 1 # training loss weight
    # Importance of the loss

    # 假设每个agent和其它的agent通信的总概率为1.
    F = half_randn(N, N) 
    # for i in range(N):
    #     F[i,i] = 0
    # F_row_sum = np.sum(F, axis=1, keepdims=True)
    # F_row_sum = np.tile(F_row_sum, (1, N))
    # F = F / F_row_sum

    # Scale of the transmission cost with L
    T = half_randn(N, N)
    T_scale = np.array([(i+1)/L for i in range(L)])

    # Loss coefficient
    load_coeff_path = '/workspace0/jhu2/Projects/ZFFT_supsup/z_paper_evaluations/data/coefficient_mat.mat'
    load_data = loadmat(load_coeff_path)
    functional_coefficient = load_data['coefficient_mat']
    prediction_mat = load_data['prediction_mat']
    step_vec = load_data['step_vec']

    loss_func_a = np.zeros((L, 1))
    loss_func_b = np.zeros((L, 1))
    loss_func_c = np.zeros((L, 1))
    loss_func_bound = np.zeros((L, 1))

    for l in range(L):
        loss_func_a[l] = functional_coefficient[l, 0]
        loss_func_b[l] = functional_coefficient[l, 1]
        loss_func_c[l] = functional_coefficient[l, 2]
        loss_func_bound[l] = functional_coefficient[l, 3]
    loss_pred_mat = prediction_mat
    loss_step_vec = step_vec
    eta_4 = 1/step_vec.shape[1]  # training loss weight
    
    CA = (half_randn()) * np.array([(i+1)/L for i in range(L)])
    k = 0
   
    # sort CA with ascending order
    num_constraints = 2*N*(N-1)*L+2*N*(N-1)**2*L+2*N*(N-1) + 2*L
    num_variables = N*(N-1)*L+N*L+2*N*(N-1)**2*L
    problem = {
        "seed": seed,
        "N": N,
        "L": L,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta_3": eta_3,
        "eta_4": eta_4,
        "F": F,
        "T": T,
        "T_scale": T_scale,
        "CA": CA,
        "k": k,
        "num_constraints": num_constraints,
        "num_variables": num_variables,
        "loss_func_a": loss_func_a,
        "loss_func_b": loss_func_b,
        "loss_func_c": loss_func_c,
        "loss_func_bound": loss_func_bound,
        "loss_pred_mat": loss_pred_mat,
        "loss_step_vec": loss_step_vec,
        "prediction_mat": prediction_mat,
        "step_vec": step_vec
    }
    print(f"[!!!] N = {N}, L = {L} problem, #constraints: {num_constraints}, #variables: {num_variables}")
    return problem



def formal_random_problem_reduced(seed=0, N=3, L=2, data_name=None, flag_multi_level_knowledge = False):
    np.random.seed(seed)
    eta_1 = 1 # loss weight
    eta_2 = 0.1 # storage cost weight
    eta_3 = 0.5 # transmission cost weight

    # 假设每个agent和其它的agent通信的总概率为1.
    F = np.random.dirichlet(np.ones(N), size=N)

    # Scale of the transmission cost with L
    # T = np.random.exponential(scale=1, size=(N, N))
    T = 1 /  np.random.lognormal(mean=0, sigma=1, size=(N, N))

    # Loss coefficient
    
    if flag_multi_level_knowledge:
        load_coeff_path = f'/workspace0/jhu2/Projects/ZFFT_supsup/z_paper_evaluations/data/task_min_loss/{data_name}_min_loss.mat'
        load_data = loadmat(load_coeff_path)
        Loss = load_data['loss_mat']
    else:
        load_coeff_path = f'/workspace0/jhu2/Projects/ZFFT_supsup/matlab_code/results/eval5_2_100epochs/{data_name}_min_loss.mat'
        load_data = loadmat(load_coeff_path)
        Loss = load_data['min_loss']
        
    Loss = Loss.flatten()
    assert len(Loss) >= L
    Loss = Loss[:L]
    
    if flag_multi_level_knowledge:
        T_scale = np.ones((L,))
        CA = np.ones((L,))
    else:
        T_scale = np.array([(i+1)/L for i in range(L)])
        CA = np.array([(i+1)/L for i in range(L)])
    
    k = 0
   
    # sort CA with ascending order
    num_constraints = 2*N*(N-1)*L+2*N*(N-1)**2*L+2*N*(N-1) + 2*L
    num_variables = N*(N-1)*L+N*L+2*N*(N-1)**2*L

    if flag_multi_level_knowledge:
        num_constraints += N*(N-1)*(L+1)*L
        num_variables += N*L
    
    problem = {
        "seed": seed,
        "N": N,
        "L": L,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta_3": eta_3,
        "F": F,
        "T": T,
        "T_scale": T_scale,
        "CA": CA,
        "k": k,
        "num_constraints": num_constraints,
        "num_variables": num_variables,
        "Loss": Loss,
        "data_name": data_name
    }
    print(f"[!!!] N = {N}, L = {L} problem, #constraints: {num_constraints}, #variables: {num_variables}")
    return problem


def subProblem(c, dim_x, A, b_cur):
    sub = gp.Model("secondary")
    sub.setParam('OutputFlag', 0)

    x = sub.addVars(dim_x, lb=0, name="x")

    # 创建约束条件
    constraints = []
    for i in range(len(b_cur)):
        coef = {j: A[i,j] for j in range(dim_x)}
        constraints.append(sub.addConstr(gp.quicksum(A[i,j] * x[j] for j in range(dim_x)) >= b_cur[i]))

    # 设置目标函数
    sub.setObjective(gp.quicksum(c[j] * x[j] for j in range(dim_x)), GRB.MINIMIZE)

    start_time = time.time()
    sub.optimize()
    end_time = time.time()
    print(f"subProblem 运行时间: {end_time - start_time} 秒")

    if sub.status == GRB.OPTIMAL:
        # 获取对偶变量值
        pi = [constr.Pi for constr in constraints]
        x_val = [x[j].X for j in range(dim_x)]
        return sub.ObjVal, pi, x_val
    else:
        raise ValueError("Subproblem is not optimal, could not happen!")


def solve_master(master, zeta, y, new_cut_E, new_cut_e):
    dim_y = len(y)
    master.addConstr(zeta - gp.quicksum(new_cut_E[i] * y[i] for i in range(dim_y)) >= new_cut_e)
    master.optimize()

    if master.status == GRB.OPTIMAL:
        return master, master.ObjVal, y
    else:
        raise ValueError("Master problem is not optimal, could not happen!")


def setupMasterProblemModel(dim_y, B_y, b_y):
    master = gp.Model("master")
    zeta = master.addVar(lb=0,vtype=GRB.CONTINUOUS, name='zeta') # zeta is the objective value of the master problem, indicate the lower bound    
    y = master.addVars(dim_y, vtype=GRB.BINARY, name='y')
    
    # 正确设置y的初始值
    for i in range(dim_y):
        y[i].Start = 1
        
    master.setObjective(zeta, sense=GRB.MINIMIZE)
    master.update()
    master.setParam('OutputFlag', 0)
    master.setParam('TimeLimit', 300)
    master.setParam('lazyConstraints', 1) # 允许在求解过程中根据需要动态添加约束

    if B_y.shape[0] > 0:
        for i in range(B_y.shape[0]):
            master.addConstr(gp.quicksum(B_y[i, j] * y[j] for j in range(dim_y)) >= b_y[i])
    
    return y, zeta, master

def generate_random_y(dim_y, problem_dict):
    k = problem_dict["k"]
    y = np.random.randint(0, 2, dim_y)
    y[k] = 1
    return y


def solveGRB_Benders_benchmark(c, d, A, B, b, B_y, b_y, eps=1e-6, max_iter=1000, mproblem_dict=None):
    '''
    Solve min cx+dy, Ax+By>=b, x \in [0,1]^N, y in {0,1}^M
    '''
    tol = eps
    iteration = 0
    assert len(d) == B.shape[1]
    
    dim_y = len(d)
    dim_x = len(c)
    y_vars, zeta_var, master = setupMasterProblemModel(dim_y, B_y, b_y)

    # 初始化y_cur为全1数组
    y_cur = np.ones(dim_y)
    cur_obj = 0
    best_obj = float("inf")

    upperbound_list = []
    gap = float("inf")

    while gap > tol and iteration < max_iter:
        b_cur = b - B @ y_cur
        
        objVal, u, x_sub = subProblem(c, dim_x, A, b_cur)
        zeta_cur = np.dot(d, y_cur) + objVal
        x_sub = np.array(x_sub)

        start_time = time.time()
        y_uint8 = y_cur.astype(np.uint8)
        x_opt = my_subproblem_solver_speed(y_uint8, mproblem_dict)
        end_time = time.time()
        x_opt_obtain_time = end_time - start_time

        hand_opt = np.dot(c, x_opt)
        machine_opt = np.dot(c, x_sub)
        to_observe = np.array([x_opt, x_sub])
        print(f"hand_opt: {hand_opt}, machine_opt: {machine_opt}")

        # 检测是否满足 Ax <= b_cur
        A_x_opt = A @ x_opt
        A_x_sub = A @ x_sub
        if np.all(A_x_opt >= b_cur):
            print("hand_opt 满足 Ax >= b_cur")
        else:
            print("hand_opt 不满足 Ax >= b_cur")
        if np.all(A_x_sub >= b_cur):
            print("machine_opt 满足 Ax >= b_cur")
        else:
            print("machine_opt 不满足 Ax <= b_cur")
        
        print(f"my_subproblem_solver 运行时间: {x_opt_obtain_time} 秒")

        y_dual = solving_dual_solution(A, b_cur, c, x_opt, x_opt_obtain_time)
        u = np.array(u)
        difference = np.abs(y_dual - u)
        if np.all(difference < 1e-6):
            print("y_dual 和 u 相等")
        else:
            print("y_dual 和 u 不相等")
            indicator = np.where(difference > 1e-6)
            relative_error = difference[indicator] / u[indicator]
            print(f"y_dual 和 u 不相等的位置: {indicator}")
            print(f"相对误差: {relative_error}")

            u_result_obj = np.dot(b_cur, u)
            print(f"u_result_obj: {u_result_obj}")
            our_result_obj = np.dot(b_cur, y_dual)
            print(f"our_result_obj: {our_result_obj}") 

        cur_obj = zeta_cur
        upperbound_list.append((cur_obj, x_sub, y_cur))

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_x = x_sub
            best_y = y_cur.copy()

        cutE = np.transpose(d - np.transpose(B) @ u)
        cute = np.dot(b, u)
        master, master_obj, y_vars = solve_master(master, zeta_var, y_vars, cutE, cute)
        
        # 在master问题求解后更新y_cur
        y_cur = np.array([y_vars[i].X for i in range(dim_y)])
        
        gap = abs(best_obj - master_obj)

        print(f"Iteration {iteration}: best upper bound = {best_obj}, lower bound = {master_obj}, gap = {gap}")
        iteration += 1

    return best_obj, best_x, best_y

def my_subproblem_solver_speed(y_cur, problem_dict):
    return my_subproblem_solver_cy(y_cur, problem_dict)
    # return my_subproblem_solver(y_cur, problem_dict)


def my_obtain_x_opt(y_cur, mproblem_dict):
    start_time = time.time()
    x_opt = my_subproblem_solver_cy(y_cur, mproblem_dict)
    end_time = time.time()
    return x_opt, end_time - start_time

def generate_problem(arguments, flag_bender_decompose=False, flag_intact_MIP=False, no_need_integrate_upbound=False, flag_too_big_for_A=False, flag_only_cd = False):
    """
    生成标准的 min cx + dy, s.t. Ax + By >= b, x in [0,1]^N, y in {0,1}^M 的问题格式。
    """
    N = arguments["N"]
    eta_1 = arguments["eta_1"]
    eta_2 = arguments["eta_2"]
    eta_3 = arguments["eta_3"]
    L = arguments["L"]
    F = arguments["F"]
    T = arguments["T"]
    T_scale = arguments["T_scale"]
    Loss = arguments["Loss"]
    CA = arguments["CA"]
    k = arguments["k"]
    num_constraints = arguments["num_constraints"]
    num_variables = arguments["num_variables"]
    
    parameter_name_x = []
    parameter_name_y = []
    parameter_dict_x = {}
    parameter_dict_y = {}
    parameter_index_x = 0
    parameter_index_y = 0

    obj_coeff = []
    c = []
    d = []

    for i in range(N):
        for j in range(N):
            if j == i: continue

            for l in range(L):
                parameter_name_x.append(f"e_i{i}_j{j}_l{l}")
                parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
                parameter_index_x += 1
                obj_coeff.append(F[i, j] * eta_1 * Loss[l] )
                c.append(obj_coeff[-1])
    
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == i: continue
                    parameter_name_x.append(f"phi_h{h}_i{i}_j{j}_l{l}")
                    parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
                    parameter_index_x += 1
                    obj_coeff.append(F[i, j] * eta_3 * T[h,i] * T_scale[l] )
                    c.append(obj_coeff[-1])
    
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == j: continue
                    parameter_name_x.append(f"vphi_h{h}_i{i}_j{j}_l{l}")
                    parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
                    parameter_index_x += 1
                    obj_coeff.append(F[i, j] * eta_3 * T[h,j] * T_scale[l] )
                    c.append(obj_coeff[-1])

    for i in range(N):
        for l in range(L):
            parameter_name_y.append(f"a_i{i}_l{l}")
            parameter_dict_y[parameter_name_y[-1]] = parameter_index_y
            parameter_index_y += 1
            obj_coeff.append(eta_2 * CA[l])
            d.append(obj_coeff[-1])

    c = np.array(c, dtype=np.float32)
    d = np.array(d, dtype=np.float32)
    
    if flag_too_big_for_A or flag_only_cd:
        # N 太大， A 太大， 无法分配存储空间 , 或者只需要更新cd
        return None, None, None, c, d, None, None, parameter_dict_x, parameter_dict_y, parameter_name_x, parameter_name_y

    num_parameter_x = parameter_index_x
    num_parameter_y = parameter_index_y
    if flag_bender_decompose:
        if not no_need_integrate_upbound:
            num_constraints = num_constraints + num_parameter_x - 2*L
        else:
            num_constraints = num_constraints - 2*L
        A = np.zeros((num_constraints, num_parameter_x), dtype=np.int8)
        B = np.zeros((num_constraints, num_parameter_y), dtype=np.int8)
        b = np.zeros((num_constraints, 1), dtype=np.int8)
        B_y = np.zeros((2*L, num_parameter_y), dtype=np.int8)
        b_y = np.zeros((2*L, 1), dtype=np.int8)
    elif flag_intact_MIP:
        num_constraints = num_constraints
        A = np.zeros((num_constraints, num_parameter_x), dtype=np.int8)
        B = np.zeros((num_constraints, num_parameter_y), dtype=np.int8)
        b = np.zeros((num_constraints, 1), dtype=np.int8)
        B_y = np.zeros((0, num_parameter_y), dtype=np.int8)
        b_y = np.zeros((0, 1), dtype=np.int8)
    
    # 约束条件1
    index_A = 0
    index_B = 0
    index_b = 0
    index_B_y = 0
    index_b_y = 0
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                # constraint_x = np.zeros(num_parameter_x)
                # constraint_y = np.zeros(num_parameter_y)
                for h in range(N):
                    if h == i: continue
                    tar_param = f"phi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = +1
                    # constraint_x[tar_idx] = -1

                tar_param = f"a_i{i}_l{l}"
                tar_idx = parameter_dict_y[tar_param]
                B[index_B, tar_idx] = +1
                # constraint_y[tar_idx] = -1


                tar_param = f"e_i{i}_j{j}_l{l}"
                tar_idx = parameter_dict_x[tar_param]
                # constraint_x[tar_idx] = 1
                A[index_A, tar_idx] = -1

                # A[index_A, :] = constraint_x
                # B[index_B, :] = constraint_y
                b[index_b, :] = 0
                index_A += 1
                index_B += 1
                index_b += 1
    
    # 约束条件2
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                # constraint_x = np.zeros(num_parameter_x)
                # constraint_y = np.zeros(num_parameter_y)
                for h in range(N):
                    if h == j: continue
                    tar_param = f"vphi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    # constraint_x[tar_idx] = -1
                    A[index_A, tar_idx] = +1

                tar_param = f"a_i{j}_l{l}"
                tar_idx = parameter_dict_y[tar_param]
                # constraint_y[tar_idx] = -1
                B[index_B, tar_idx] = +1

                tar_param = f"e_i{i}_j{j}_l{l}"
                tar_idx = parameter_dict_x[tar_param]
                # constraint_x[tar_idx] = 1
                A[index_A, tar_idx] = -1

                # A[index_A, :] = constraint_x
                # B[index_B, :] = constraint_y
                b[index_b, :] = 0
                index_A += 1
                index_B += 1
                index_b += 1

    # 约束条件3
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == i: continue
                    # constraint_x = np.zeros(num_parameter_x)
                    # constraint_y = np.zeros(num_parameter_y)
                    tar_param = f"phi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    # constraint_x[tar_idx] = 1
                    A[index_A, tar_idx] = -1

                    tar_param = f"a_i{h}_l{l}"
                    tar_idx = parameter_dict_y[tar_param]
                    # constraint_y[tar_idx] = -1
                    B[index_B, tar_idx] = +1

                    # A[index_A, :] = constraint_x
                    # B[index_B, :] = constraint_y
                    b[index_b, :] = 0
                    index_A += 1
                    index_B += 1
                    index_b += 1
    # 约束条件4
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == j: continue
                    # constraint_x = np.zeros(num_parameter_x)
                    # constraint_y = np.zeros(num_parameter_y)
                    tar_param = f"vphi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    # constraint_x[tar_idx] = 1
                    A[index_A, tar_idx] =-1

                    tar_param = f"a_i{h}_l{l}"
                    tar_idx = parameter_dict_y[tar_param]
                    # constraint_y[tar_idx] = -1
                    B[index_B, tar_idx] = +1

                    # A[index_A, :] = constraint_x
                    # B[index_B, :] = constraint_y
                    b[index_b, :] = 0
                    index_A += 1
                    index_B += 1
                    index_b += 1
    # 约束条件5
    for i in range(N):
        for j in range(N):
            if j == i: continue
            # constraint_x_1 = np.zeros(num_parameter_x)
            # constraint_x_2 = np.zeros(num_parameter_x)
            # constraint_y_1 = np.zeros(num_parameter_y)
            # constraint_y_2 = np.zeros(num_parameter_y)
            for l in range(L):
                tar_param = f"e_i{i}_j{j}_l{l}"
                tar_idx = parameter_dict_x[tar_param]
                # constraint_x_1[tar_idx] = 1
                A[index_A, tar_idx] = 1
                # constraint_x_2[tar_idx] = -1
                A[index_A + 1, tar_idx] = -1

            # A[index_A, :] = constraint_x_1
            # A[index_A + 1, :] = constraint_x_2
            # B[index_B, :] = constraint_y_1
            # B[index_B + 1, :] = constraint_y_2
            b[index_b, :] = 1
            b[index_b + 1, :] = -1
            index_A += 2
            index_B += 2
            index_b += 2
    
    if flag_bender_decompose:
        for l in range(L):
            # constraint_y_1 = np.zeros(num_parameter_y)
            # constraint_y_2 = np.zeros(num_parameter_y)
            tar_param = f"a_i{k}_l{l}"
            tar_idx = parameter_dict_y[tar_param]
            # constraint_y_1[tar_idx] = 1
            B_y[index_B_y, tar_idx] = 1
            # constraint_y_2[tar_idx] = -1
            B_y[index_B_y + 1, tar_idx] = -1
            # B_y[index_B_y, :] = constraint_y_1
            # B_y[index_B_y + 1, :] = constraint_y_2
            b_y[index_b_y, :] = 1
            b_y[index_b_y + 1, :] = -1
            index_B_y += 2
            index_b_y += 2
    elif flag_intact_MIP:
        for l in range(L):
            # constraint_x = np.zeros(num_parameter_x)
            # constraint_y_1 = np.zeros(num_parameter_y)
            # constraint_y_2 = np.zeros(num_parameter_y)
            tar_param = f"a_i{k}_l{l}"
            tar_idx = parameter_dict_y[tar_param]
            # constraint_y_1[tar_idx] = 1
            B[index_B, tar_idx] = 1
            # constraint_y_2[tar_idx] = -1
            B[index_B + 1, tar_idx] = -1

            # A[index_A, :] = constraint_x
            # A[index_A + 1, :] = constraint_x
            # B[index_B, :] = constraint_y_1
            # B[index_B + 1, :] = constraint_y_2
            b[index_b, :] = 1
            b[index_b + 1, :] = -1
            index_A += 2
            index_B += 2
            index_b += 2

    if flag_bender_decompose and not no_need_integrate_upbound:
        #NOTE - 其实这里不需要加入上限条件，因为phi<a这个条件默认包含了上限。但是这里写上有助于提高增幅的对比。
        for i in range(num_parameter_x):
            # constraint_x = np.zeros(num_parameter_x)
            # constraint_x[i] = 1
            A[index_A, i] = -1
            # constraint_y = np.zeros(num_parameter_y)
            # A[index_A, :] = constraint_x
            # B[index_B, :] = constraint_y
            b[index_b, :] = -1
            index_A += 1
            index_B += 1
            index_b += 1

    assert A.shape[0] == B.shape[0] == b.shape[0]
    print("Total number of parameters: ", num_parameter_x + num_parameter_y)
    print("Total number of constraints: ", A.shape[0])

    # # 用黑白点阵图画出A的稀疏性并保存下来
    # plt.figure(figsize=(10, 10))
    # plt.imshow(A, cmap='gray')
    # plt.colorbar()
    # plt.savefig("./A.png")
    # plt.show()

    b = b.squeeze()
    return A, B, b, c, d, B_y, b_y, parameter_dict_x, parameter_dict_y, parameter_name_x, parameter_name_y


def generate_problem_multi_level_know(arguments, flag_too_big_for_A=False, flag_only_cd = False, flag_k_save_all=True):
    """
    生成标准的 min cx + dy, s.t. Ax + By >= b, x in [0,1]^N, y in {0,1}^M 的问题格式。
    来源于 generate_problem 函数，多加了多级知识的部分。
    """

    N = arguments["N"]
    eta_1 = arguments["eta_1"]
    eta_2 = arguments["eta_2"]
    eta_3 = arguments["eta_3"]
    L = arguments["L"]
    F = arguments["F"]
    T = arguments["T"]
    T_scale = arguments["T_scale"]
    Loss = arguments["Loss"]
    CA = arguments["CA"]
    k = arguments["k"]

    
    num_constraints = arguments["num_constraints"]
    num_variables = arguments["num_variables"]
    
    parameter_name_x = []
    parameter_name_y = []
    parameter_dict_x = {}
    parameter_dict_y = {}
    parameter_index_x = 0
    parameter_index_y = 0

    obj_coeff = []
    c = []
    d = []

    for i in range(N):
        for j in range(N):
            if j == i: continue

            for l in range(L):
                parameter_name_x.append(f"e_i{i}_j{j}_l{l}")
                parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
                parameter_index_x += 1
                obj_coeff.append(F[i, j] * eta_1 * Loss[l] )
                c.append(obj_coeff[-1])
    
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == i: continue
                    parameter_name_x.append(f"phi_h{h}_i{i}_j{j}_l{l}")
                    parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
                    parameter_index_x += 1
                    obj_coeff.append(F[i, j] * eta_3 * T[h,i] * T_scale[l] )
                    c.append(obj_coeff[-1])
    
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == j: continue
                    parameter_name_x.append(f"vphi_h{h}_i{i}_j{j}_l{l}")
                    parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
                    parameter_index_x += 1
                    obj_coeff.append(F[i, j] * eta_3 * T[h,j] * T_scale[l] )
                    c.append(obj_coeff[-1])

    for i in range(N):
        for l in range(L):
            parameter_name_x.append(f"tau_i{i}_l{l}") # tau_i_l 代表 agent i 是否使用到了 level l 的知识。
            parameter_dict_x[parameter_name_x[-1]] = parameter_index_x
            parameter_index_x += 1
            obj_coeff.append(0) # 其本身并不对目标函数产生影响。
            c.append(obj_coeff[-1])

    for i in range(N):
        for l in range(L):
            parameter_name_y.append(f"a_i{i}_l{l}")
            parameter_dict_y[parameter_name_y[-1]] = parameter_index_y
            parameter_index_y += 1
            obj_coeff.append(eta_2 * CA[l])
            d.append(obj_coeff[-1])


    c = np.array(c, dtype=np.float32)
    d = np.array(d, dtype=np.float32)
    
    if flag_too_big_for_A or flag_only_cd:
        # N 太大， A 太大， 无法分配存储空间 , 或者只需要更新cd
        return None, None, None, c, d

    num_parameter_x = parameter_index_x
    num_parameter_y = parameter_index_y
    
    A = np.zeros((num_constraints, num_parameter_x), dtype=np.int8)
    B = np.zeros((num_constraints, num_parameter_y), dtype=np.int8)
    b = np.zeros((num_constraints, 1), dtype=np.int8)

    # 约束条件1
    index_A = 0
    index_B = 0
    index_b = 0
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == i: continue
                    tar_param = f"phi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = +1

                tar_param = f"a_i{i}_l{l}"
                tar_idx = parameter_dict_y[tar_param]
                B[index_B, tar_idx] = +1

                tar_param = f"tau_i{i}_l{l}"
                tar_idx = parameter_dict_x[tar_param]
                A[index_A, tar_idx] = -1

                b[index_b, :] = 0
                index_A += 1
                index_B += 1
                index_b += 1
    
    # 约束条件2
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == j: continue
                    tar_param = f"vphi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = +1

                tar_param = f"a_i{j}_l{l}"
                tar_idx = parameter_dict_y[tar_param]
                B[index_B, tar_idx] = +1

                tar_param = f"tau_i{j}_l{l}"
                tar_idx = parameter_dict_x[tar_param]
                A[index_A, tar_idx] = -1

                b[index_b, :] = 0
                index_A += 1
                index_B += 1
                index_b += 1

    # 约束条件3&4
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for l_1 in range(l,L):
                    tar_param = f"tau_i{i}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = +1

                    tar_param = f"e_i{i}_j{j}_l{l_1}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = -1

                    b[index_b, :] = 0
                    index_A += 1
                    index_B += 1
                    index_b += 1

                    tar_param = f"tau_i{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = +1

                    tar_param = f"e_i{i}_j{j}_l{l_1}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = -1

                    b[index_b, :] = 0
                    index_A += 1
                    index_B += 1
                    index_b += 1

    

    # 约束条件5
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == i: continue
                    tar_param = f"phi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] = -1

                    tar_param = f"a_i{h}_l{l}"
                    tar_idx = parameter_dict_y[tar_param]
                    B[index_B, tar_idx] = +1

                    b[index_b, :] = 0
                    index_A += 1
                    index_B += 1
                    index_b += 1
    # 约束条件6
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                for h in range(N):
                    if h == j: continue
                    tar_param = f"vphi_h{h}_i{i}_j{j}_l{l}"
                    tar_idx = parameter_dict_x[tar_param]
                    A[index_A, tar_idx] =-1

                    tar_param = f"a_i{h}_l{l}"
                    tar_idx = parameter_dict_y[tar_param]
                    B[index_B, tar_idx] = +1

                    b[index_b, :] = 0
                    index_A += 1
                    index_B += 1
                    index_b += 1
    # 约束条件7
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                tar_param = f"e_i{i}_j{j}_l{l}"
                tar_idx = parameter_dict_x[tar_param]
                A[index_A, tar_idx] = 1
                A[index_A + 1, tar_idx] = -1

            b[index_b, :] = 1
            b[index_b + 1, :] = -1
            index_A += 2
            index_B += 2
            index_b += 2
    
    if flag_k_save_all: # 拥有知识的agent 需要保存所有知识
        for l in range(L):
            tar_param = f"a_i{k}_l{l}"
            tar_idx = parameter_dict_y[tar_param]
            B[index_B, tar_idx] = 1
            B[index_B + 1, tar_idx] = -1

            b[index_b, :] = 1
            b[index_b + 1, :] = -1
            index_A += 2
            index_B += 2
            index_b += 2
    
    assert A.shape[0] == B.shape[0] == b.shape[0]
    assert A.shape[0] == num_constraints

    print("Total number of parameters: ", num_parameter_x + num_parameter_y)
    print("Total number of constraints: ", A.shape[0])

    b = b.squeeze()
    return A, B, b, c, d

def solve_full_mip_scipy(c, d, A, B, b, mproblem_dict=None):
    """
    使用 scipy.optimize.milp 求解完整的混合整数规划问题
    min cx + dy
    s.t. Ax + By >= b
         x in [0,1]^N
         y in {0,1}^M
    """
    N = len(c)
    M = len(d)
    
    # 合并决策变量和目标系数
    c_full = np.concatenate([c, d])
    
    # 合并约束矩阵
    A_full = np.hstack([A, B])
    
    # 设置变量的界限
    # x 和 y 都是二进制变量，所以上下界都是 0 和 1
    lb = np.zeros(N + M)  # 下界
    ub = np.ones(N + M)   # 上界
    bounds = Bounds(lb, ub)
    
    # 设置整数变量的索引（所有变量都是整数）
    integrality = np.ones(N + M)
    
    constraints = LinearConstraint(A_full, lb=b)
    # start_time = time.time()
    result = milp(
        c=c_full,            # 目标函数系数
        constraints=constraints,
        bounds=bounds,       # 变量界限
        integrality=integrality, # 整数约束
        options={"disp": False}  # 不显示求解过程
    )
    # end_time = time.time()
    # print(f"solve_full_mip_scipy 运行时间: {end_time - start_time} 秒")



    if result.success:
        # 分离结果中的 x 和 y
        x_val = result.x[:N]
        y_val = result.x[N:]
        # y_uint8 = np.array(y_val, dtype=np.uint8)
        # x_opt = my_subproblem_solver_cy(y_uint8, mproblem_dict)
        # if not np.all(np.isclose(x_opt, x_val, atol=1e-6)):
        #     print("x_opt 和 x_val 不相等")
        #     obj_x_val = np.sum(x_val*c)
        #     obj_x_opt = np.sum(x_opt*c)
        #     print(f"obj_x_val: {obj_x_val}, obj_x_opt: {obj_x_opt}")
        obj_val = result.fun
        # print(f"MIP求解结果 (scipy): {obj_val}")
        ka_loss, trans_loss, store_loss = calculate_three_part_loss(x_val, y_val, c, d, mproblem_dict)
        total_loss = ka_loss + trans_loss + store_loss
        print(f"[!!] Ratios: ka: {ka_loss/total_loss:.2f}, trans: {trans_loss/total_loss:.2f}, store: {store_loss/total_loss:.2f}")
        return obj_val, x_val.tolist(), y_val.tolist()
    else:
        raise ValueError("Scipy MIP求解失败")

def solve_full_ga(c, d, mproblem_dict, flag_multi_level_knowledge=False):
    import pygad
    import numpy as np

    L = mproblem_dict["L"]
    N = mproblem_dict["N"]
    def fitness_func(ga_instance, solution, solution_idx):
        to_insert = np.ones(L, dtype=np.uint8)
        y_uint8 = np.array(solution, dtype=np.uint8)
        y_uint8 = np.insert(y_uint8, 0, to_insert)
        if flag_multi_level_knowledge:
            x_opt = my_subproblem_solver_cy_MK(y_uint8, mproblem_dict)
            output = np.sum(x_opt*c[:len(x_opt)]) + np.sum(y_uint8*d)
        else:
            x_opt = my_subproblem_solver_cy(y_uint8, mproblem_dict)
            output = np.sum(x_opt*c) + np.sum(y_uint8*d)
        fitness = -output
        return fitness

    n_var = (mproblem_dict["N"]-1) * mproblem_dict["L"]
    
    ga_instance = pygad.GA(
        num_generations=10*N,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=10*N,
        num_genes=n_var,
        gene_space=[0, 1],  # 二进制变量
        mutation_type="random",
        mutation_percent_genes=25
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    return -solution_fitness, solution

def solve_full_mip(c, d, A, B, b, flag_soft_time_limit=False):
    """
    直接求解完整的混合整数规划问题
    min cx + dy
    s.t. Ax + By <= b
         x in [0,1]^N
         y in {0,1}^M
    """
    
    N = len(c)
    M = len(d)
    
    model = gp.Model("full_mip")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 1800)
    
    x = model.addVars(N, lb=0.0, name="x")
    y = model.addVars(M, vtype=GRB.BINARY, name="y")
    
    # 将变量转换为列表形式
    vars_x = [x[i] for i in range(N)]
    vars_y = [y[i] for i in range(M)]
    all_vars = vars_x + vars_y
    all_coeffs = np.concatenate([c, d])
    
    # 使用矩阵形式一次性设置目标函数
    model.setObjective(gp.LinExpr(all_coeffs, all_vars), GRB.MINIMIZE)

    AB = np.concatenate([A, B], axis=1)
    # 使用矩阵形式一次性添加所有约束
    model.addMConstr(AB, all_vars, '>=', b)
    
    # start_time = time.time()
    # print(f"完成条件的建设，开始求解MIP问题")
    model.optimize()
    # end_time = time.time()
    # print(f"solve_full_mip 运行时间: {end_time - start_time} 秒")
    
    if model.status == GRB.OPTIMAL:
        x_val = [x[j].X for j in range(N)]
        y_val = [y[j].X for j in range(M)]
        # print(f"MIP求解结果: {model.ObjVal}")
        return model.ObjVal, x_val, y_val
    elif model.status == GRB.TIME_LIMIT:
        print("求解超时!")
        x_val = [x[j].X for j in range(N)]
        y_val = [y[j].X for j in range(M)]
        # print(f"MIP求解结果: {model.ObjVal}")
        if flag_soft_time_limit:
            return model.ObjVal, x_val, y_val
        else:
            return model.ObjVal, "reach_time_limit", True
    else:
        raise ValueError("MIP求解失败")

def solve_my_bender_decompose(c, d, A, B, b, B_y, b_y, eps=1e-6, max_iter=1000, mproblem_dict=None, y_cur_init=None):
    tol = eps
    iteration = 0
    assert len(d) == B.shape[1]
    
    dim_y = len(d)
    dim_x = len(c)
    y_vars, zeta_var, master = setupMasterProblemModel(dim_y, B_y, b_y)

    # 初始化y_cur为全1数组
    if y_cur_init is None:
        y_cur = np.ones(dim_y)
    else:
        y_cur = y_cur_init
    cur_obj = 0
    best_obj = float("inf")

    upperbound_list = []
    gap = float("inf")

    while gap > tol and iteration < max_iter:
        b_cur = b - B @ y_cur
        y_uint8 = y_cur.astype(np.uint8)
        x_opt = my_subproblem_solver_speed(y_uint8, mproblem_dict)
        zeta_cur = np.dot(d, y_cur) + np.dot(c, x_opt)
        u = solving_dual_solution(A, b_cur, c, x_opt)
        u = np.array(u)
        
        cur_obj = zeta_cur
        upperbound_list.append((cur_obj, x_opt, y_cur))

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_x = x_opt
            best_y = y_cur.copy()

        cutE = np.transpose(d - np.transpose(B) @ u)
        cute = np.dot(b, u)
        master, master_obj, y_vars = solve_master(master, zeta_var, y_vars, cutE, cute)
        
        # 在master问题求解后更新y_cur
        y_cur = np.array([y_vars[i].X for i in range(dim_y)])
        
        gap = abs(best_obj - master_obj)

        print(f"Iteration {iteration}: best upper bound = {best_obj}, lower bound = {master_obj}, gap = {gap}")
        iteration += 1

    ka_loss, trans_loss, store_loss = calculate_three_part_loss(best_x, best_y, c, d, mproblem_dict)
    total_loss = ka_loss + trans_loss + store_loss
    print(f"[!!] Ratios: ka: {ka_loss/total_loss:.2f}, trans: {trans_loss/total_loss:.2f}, store: {store_loss/total_loss:.2f}")
    
    return best_obj, best_x, best_y

def calculate_three_part_loss(x, y, c, d, mproblem_dict):
    N = mproblem_dict["N"]
    L = mproblem_dict["L"]
    
    e_ijl_indicator = np.zeros(len(c), dtype=np.uint8)
    phi_hijl_indicator = np.zeros(len(c), dtype=np.uint8)
    vphi_hijl_indicator = np.zeros(len(c), dtype=np.uint8)

    num_e_ijl = N * (N-1) * L
    num_phi_hijl = N * (N-1)**2 * L
    num_vphi_hijl = N * (N-1)**2 * L
    
    # assert len(x) == num_e_ijl + num_phi_hijl + num_vphi_hijl

    e_ijl_indicator[:num_e_ijl] = x[:num_e_ijl]
    phi_hijl_indicator[num_e_ijl:num_e_ijl + num_phi_hijl] = x[num_e_ijl:num_e_ijl + num_phi_hijl]
    vphi_hijl_indicator[num_e_ijl + num_phi_hijl:num_e_ijl + num_phi_hijl + num_vphi_hijl] = x[num_e_ijl + num_phi_hijl:num_e_ijl + num_phi_hijl + num_vphi_hijl]

    ka_loss = np.sum(e_ijl_indicator * c)
    trans_loss = np.sum(phi_hijl_indicator * c) + np.sum(vphi_hijl_indicator * c)
    store_loss = np.sum(d * y)

    return ka_loss, trans_loss, store_loss

            
def solve_all_save(c, d, mproblem_dict, flag_multi_level_knowledge=False):
    N = mproblem_dict["N"]
    L = mproblem_dict["L"]

    y_uint8 = np.ones(N*L, dtype=np.uint8)
    if flag_multi_level_knowledge:
        x_opt = my_subproblem_solver_cy_MK(y_uint8, mproblem_dict)
        output = np.sum(x_opt*c[:len(x_opt)]) + np.sum(y_uint8*d)
    else:
        x_opt = my_subproblem_solver_cy(y_uint8, mproblem_dict)
        output = np.sum(x_opt*c) + np.sum(y_uint8*d)
    return output, None

def solve_no_save(c, d, mproblem_dict, flag_multi_level_knowledge=False):
    N = mproblem_dict["N"]
    L = mproblem_dict["L"]

    y_uint8 = np.ones(N*L, dtype=np.uint8)
    y_uint8[L:] = 0
    if flag_multi_level_knowledge:
        x_opt = my_subproblem_solver_cy_MK(y_uint8, mproblem_dict)
        output = np.sum(x_opt*c[:len(x_opt)]) + np.sum(y_uint8*d)
    else:
        x_opt = my_subproblem_solver_cy(y_uint8, mproblem_dict)
        output = np.sum(x_opt*c) + np.sum(y_uint8*d)
    return output, None

def solve_greedy(c, d, mproblem_dict, 
                 flag_multi_level_knowledge=False,
                 flag_track_iter=False, # 是否记录每轮迭代的结果
                 I_max=100, # 默认的最大的步数
                 flag_til_converge=False,
                 c_norm=None,
                 d_norm=None,
                 ):
    
    N = mproblem_dict["N"]
    L = mproblem_dict["L"]
    L_case = 2**L
    y_cur = np.zeros(N*L, dtype=np.uint8)
    y_cur[0:L] = 1
    x_best = None
    iteration_step = 0
    real_max_iter = 1e5
    I_max_iter = I_max * N
    if flag_track_iter:
        iter_result_list = []
    break_flag = False
    num_user_no_change = 0
    while True:
        for i in range(1,N):
            if iteration_step >= 1:
                temp_l_binary = y_cur[L*i:L*(i+1)]
            else:
                temp_l_binary = None
            y_cur_temp = y_cur.copy()
            best_obj = float("inf")
            for l in range(L_case):
                # 生成固定L位的二进制数
                l_binary = np.zeros(L, dtype=np.uint8)
                for bit in range(L):
                    l_binary[L-1-bit] = (l >> bit) & 1
                y_cur_temp[L*i:L*(i+1)] = l_binary
                if flag_multi_level_knowledge:
                    x_opt = my_subproblem_solver_cy_MK(y_cur_temp, mproblem_dict)
                    obj_val = np.sum(x_opt*c[:len(x_opt)]) + np.sum(y_cur_temp*d)
                else:
                    x_opt = my_subproblem_solver_cy(y_cur_temp, mproblem_dict)
                    obj_val = np.sum(x_opt*c) + np.sum(y_cur_temp*d)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_y = y_cur_temp[L*i:L*(i+1)].copy()
                    x_best = x_opt
            iteration_step += 1
            if flag_track_iter:
                iter_result_list.append(best_obj)
            if temp_l_binary is not None and np.allclose(best_y, temp_l_binary):
                num_user_no_change += 1
            else:
                num_user_no_change = 0
            if num_user_no_change >= N:
                break
            y_cur[L*i:L*(i+1)] = best_y
        if not flag_til_converge and iteration_step >= I_max_iter:
            break
        if iteration_step >= real_max_iter:
            print("Warning: Reach real max iter!")
            break
        if num_user_no_change >= N:
            print(f"Converged! Using {iteration_step} iterations")
            break
    
    if c_norm is None or d_norm is None:
        ka_loss, trans_loss, store_loss = calculate_three_part_loss(x_best, y_cur, c, d, mproblem_dict)
    else:
        ka_loss, trans_loss, store_loss = calculate_three_part_loss(x_best, y_cur, c_norm, d_norm, mproblem_dict)
    total_loss = ka_loss + trans_loss + store_loss
    loss_vec = [ka_loss, trans_loss, store_loss]
    if flag_track_iter:
        return best_obj, y_cur, loss_vec, iter_result_list
    else:
        return best_obj, y_cur, loss_vec


def solve_greedy_pruned(c, d, mproblem_dict, 
                        flag_multi_level_knowledge=False,
                        I_max=100):
    
    N = mproblem_dict["N"]
    L = mproblem_dict["L"]
    y_cur = np.zeros(N*L, dtype=np.uint8)
    y_cur[0:L] = 1
    for iteration in range(I_max):
        for i in range(1,N):
            if iteration >= 1:
                cur_best_y = y_cur[L*i:L*(i+1)]
            else:
                cur_best_y = None
            y_cur_temp = y_cur.copy()
            best_obj = float("inf")
            cur_best_y = y_cur_temp[L*i:L*(i+1)].copy()
            this_round_y = np.zeros(L, dtype=np.uint8)
            y_cur_temp[L*i:L*(i+1)] = this_round_y
            if flag_multi_level_knowledge:
                x_opt = my_subproblem_solver_cy_MK(y_cur_temp, mproblem_dict)
                obj_val = np.sum(x_opt*c[:len(x_opt)]) + np.sum(y_cur_temp*d)
            else:
                x_opt = my_subproblem_solver_cy(y_cur_temp, mproblem_dict)
                obj_val = np.sum(x_opt*c) + np.sum(y_cur_temp*d)
            best_obj = obj_val
            for cur_l in range(1,L+1):
                flag_added = False
                best_adding = None
                for consider_l in range(L):
                    # 尝试在没有的地方加一个
                    if this_round_y[consider_l] == 1: continue
                    temp_this_round_y = this_round_y.copy()
                    temp_this_round_y[consider_l] = 1
                    y_cur_temp[L*i:L*(i+1)] = temp_this_round_y
                    if flag_multi_level_knowledge:
                        x_opt = my_subproblem_solver_cy_MK(y_cur_temp, mproblem_dict)
                        obj_val = np.sum(x_opt*c[:len(x_opt)]) + np.sum(y_cur_temp*d)
                    else:
                        x_opt = my_subproblem_solver_cy(y_cur_temp, mproblem_dict)
                        obj_val = np.sum(x_opt*c) + np.sum(y_cur_temp*d)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_adding = temp_this_round_y
                        flag_added = True
                if not flag_added: break
                this_round_y = best_adding
            if cur_best_y is not None and np.allclose(this_round_y, cur_best_y):
                break
            y_cur[L*i:L*(i+1)] = this_round_y
    return best_obj, y_cur


def prepare_dict_for_my_solver(problem_dict):
    L = problem_dict["L"]
    N = problem_dict["N"]
    T = problem_dict["T"]
    T_scale = problem_dict["T_scale"]
    T_ = np.expand_dims(T, axis = 2)
    T_scaled = np.tile(T_, (1,1,L)) * np.tile(np.transpose(T_scale), (N,N,1))
    eta_1 = problem_dict["eta_1"]
    Loss = problem_dict["Loss"]
    eta_1_Loss = eta_1 * Loss # (L,)
    eta_1_Loss = np.tile(eta_1_Loss, (N,N,1)) # 这个其实不用每次都计算。可以放在函数外计算。
    mproblem_dict = {
        "T_scaled": T_scaled,
        "eta_1_Loss": eta_1_Loss,
        "eta_3": problem_dict["eta_3"],
        "N": problem_dict["N"],
        "L": problem_dict["L"],
    }
    return mproblem_dict

def obtain_coeff_for_loss(x_val, c, N, L):
    coeff_loss = np.zeros((L,))
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                j_adjust = j if j < i else j - 1
                index = i * (N-1) * L + j_adjust * L + l
                coeff_loss[l] += x_val[index] * c[index]
    return coeff_loss

def solve_complete_problem_with_training_loss(problem, max_iter=500):
    N = problem["N"]
    L = problem["L"]
    eta_4 = problem["eta_4"]
    loss_func_a = problem["loss_func_a"]
    loss_func_b = problem["loss_func_b"]
    loss_func_c = problem["loss_func_c"]
    loss_func_bound = problem["loss_func_bound"]
    loss_step_vec = problem["loss_step_vec"]
    loss_pred_mat = problem["loss_pred_mat"]
    num_steps = loss_step_vec.shape[1]

    def get_gradient_vec(t_a, t_b, t_vec):
        gradient_vec = t_a * t_b * np.power(t_vec, t_b - 1)
        return gradient_vec

    gradient_mat = np.zeros((L, num_steps))
    for l in range(L):
        gradient_mat[l,:] = get_gradient_vec(loss_func_a[l], loss_func_b[l], loss_step_vec)


    add_training_steps = np.zeros((L,))

    Loss = loss_pred_mat[:,0]
    for iter in range(max_iter):
        problem["Loss"] = Loss
        A, B, b, c, d, B_y, b_y, _, _, _, _ = \
                    generate_problem(problem, flag_intact_MIP=True)
        obj_val, x_val, y_val = solve_full_mip_scipy(c, d, A, B, b, mproblem_dict=problem)
        coeff_loss = obtain_coeff_for_loss(x_val, c, N, L)

        sum_obj_val = obj_val + eta_4 * np.sum(add_training_steps)
        print(f"iter: {iter}, sum_obj_val: {sum_obj_val}, add_training_steps: {add_training_steps}")
        # 找到gradient_mat里，满足大于 小于 负eta_4的最大的index
        add_training_steps_old = add_training_steps.copy()
        add_training_steps = np.zeros((L,))
        metric = np.zeros((L,num_steps))
        for l in range(L):
            for t in range(num_steps):
                metric[l,t] = gradient_mat[l,t] * coeff_loss[l]
                if metric[l,t] < -eta_4:
                    add_training_steps[l] = t
                    Loss[l] = loss_pred_mat[l, t]
        if np.allclose(add_training_steps, add_training_steps_old):
            print("Converged!")
            break
        
        print(f"obj_val: {obj_val}")

    ka_loss, trans_loss, store_loss = calculate_three_part_loss(x_val, y_val, c, d, problem)
    train_loss = np.sum(add_training_steps)     
    total_loss = ka_loss + trans_loss + store_loss + train_loss

    left = A @ x_val + B @ y_val - b
    if np.any(left < -1e-6):
        print("Condition violated!")

    check_condition(N, L, x_val, y_val)
    y_val = np.reshape(y_val, (N,L))
    print(f"allocation: {y_val}")
    print(f"percentage: ka: {ka_loss/total_loss:.6f}")
    print(f"transmission: {trans_loss/total_loss:.6f}")
    print(f"store: {store_loss/total_loss:.6f}")
    print(f"training: {train_loss/total_loss:.6f}")

    # 计算一个comparison case
    Loss = loss_pred_mat[:,-1]
    problem["Loss"] = Loss
    A, B, b, c, d, B_y, b_y, _, _, _, _ = \
                generate_problem(problem, flag_intact_MIP=True)
    obj_val, x_val, y_val = solve_full_mip_scipy(c, d, A, B, b, mproblem_dict=problem)
    coeff_loss = obtain_coeff_for_loss(x_val, c, N, L)

    sum_obj_val_comparison = obj_val + eta_4 * num_steps * L

    print(f"ratio: {sum_obj_val/sum_obj_val_comparison:.2f}")


def solve_reduced_problem(problem, standard_prob_element, flag_time=False, flag_soft_time_limit=False, c_norm=None, d_norm=None):
    N = problem["N"]
    L = problem["L"]
    Loss = problem["Loss"]
    eta_1 = problem["eta_1"]
    eta_2 = problem["eta_2"]
    eta_3 = problem["eta_3"]

    A = standard_prob_element["A"]
    B = standard_prob_element["B"]
    b = standard_prob_element["b"]
    c = standard_prob_element["c"]
    d = standard_prob_element["d"]

    if flag_time:
        start_time = time.time()

    obj_val, x_val, y_val = solve_full_mip(c, d, A, B, b, flag_soft_time_limit=flag_soft_time_limit)

    if flag_time:
        end_time = time.time()
        used_time = end_time - start_time
    if x_val == "reach_time_limit":
        used_time = -1
    

    if False: # 这下面是测试用的，目前在跑结果，就先不用了
        used_level = summary_used_level(x_val, N, L)
        # print(f"used_level: {used_level}")
        stored_level = summary_stored_level(y_val, N, L)
        # print(f"stored_level: {stored_level}")
    else:
        used_level = None
        stored_level = None

    if not flag_time:
        if c_norm is None or d_norm is None:
            ka_loss, trans_loss, store_loss = calculate_three_part_loss(x_val, y_val, c, d, problem)
        else:
            ka_loss, trans_loss, store_loss = calculate_three_part_loss(x_val, y_val, c_norm, d_norm, problem)
        total_loss = ka_loss + trans_loss + store_loss
        loss_vec = [ka_loss, trans_loss, store_loss]
        loss_ratios = [ka_loss/total_loss, trans_loss/total_loss, store_loss/total_loss]
        print(f"optimal loss ratios: ka: {loss_ratios[0]:.2f}, trans: {loss_ratios[1]:.2f}, store: {loss_ratios[2]:.2f}")
    else:
        loss_vec = None
        loss_ratios = None

    if flag_time:
        return obj_val, used_level, stored_level, loss_vec, used_time
    else:
        return obj_val, used_level, stored_level, loss_vec


def summary_stored_level(y_val, N, L):
    y_val_ = np.reshape(y_val, (-1, L))
    stored_level = np.sum(y_val_, axis=0)
    return stored_level

def summary_used_level(x_val, N, L):
    used_level = np.zeros((L,))
    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                j_adjust = j if j < i else j - 1
                index = i * (N-1) * L + j_adjust * L + l
                if x_val[index] > 0:
                    used_level[l] += 1
    return used_level

def analysis_loss_components(problem):
    Loss = problem["Loss"]

    eta_1 = problem["eta_1"]
    eta_2 = problem["eta_2"]
    eta_3 = problem["eta_3"]
    
    mean_loss = np.mean(Loss)
    ka_loss_average = mean_loss * eta_1
    
    T = problem["T"]
    T_scale = problem["T_scale"]
    avg_T = np.mean(T) * np.mean(T_scale)
    transmission_loss_average = eta_2 * avg_T

    CA = problem["CA"]
    store_loss_average = eta_3 * np.mean(CA)


    print(f"ka_loss_average: {ka_loss_average}, transmission_loss_average: {transmission_loss_average}, store_loss_average: {store_loss_average}")
    total_loss_average = ka_loss_average + transmission_loss_average + store_loss_average   
    print(f"percentage: ka: {ka_loss_average/total_loss_average:.2f}, transmission: {transmission_loss_average/total_loss_average:.2f}, store: {store_loss_average/total_loss_average:.2f}")
    
def check_condition(N, L, x_val, y_val):
    e_ijl_indicator = x_val[:N*(N-1)*L]
    e_ijl_observe = np.reshape(e_ijl_indicator, (-1, L))
    phi_hijl_indicator = x_val[N*(N-1)*L:N*(N-1)*L + N*(N-1)**2*L]
    vphi_hijl_indicator = x_val[N*(N-1)*L + N*(N-1)**2*L:]

    for i in range(N):
        for j in range(N):
            if j == i: continue
            for l in range(L):
                j_adjust = j if j < i else j - 1
                index = i * (N-1) * L + j_adjust * L + l

                if e_ijl_indicator[index] == 1:
                    index_2 = i*L + l
                    if y_val[index_2] == 0:
                        found_transmit = False
                        for h in range(N):
                            if h == i: continue
                            h_adjust = h if h < i else h - 1
                            index_3 = i*(N-1)**2*L + j_adjust*(N-1)*L + l * (N-1) + h_adjust
                            if np.isclose(phi_hijl_indicator[index_3], 1):
                                found_transmit = True
                                break
                        if not found_transmit:
                            print("Condition violated!")
                    index_2 = j*L + l
                    if y_val[index_2] == 0:
                        found_store = False
                        for h in range(N):
                            if h == j: continue
                            h_adjust = h if h < j else h - 1
                            index_3 = i*(N-1)**2*L + j_adjust*(N-1)*L + l * (N-1) + h_adjust
                            if np.isclose(vphi_hijl_indicator[index_3], 1):
                                found_store = True
                                break
                        if not found_store:
                            print("Condition violated!")
                            

