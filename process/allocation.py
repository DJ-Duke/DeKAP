from datetime import datetime
import time
import numpy as np
from scipy.io import savemat
import multiprocessing as mp
from functools import partial
import logging
from multiprocessing import Manager
import sys
import contextlib
import os
from multiprocessing import Manager
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"[!] Project root is now set to: {PROJECT_ROOT}")

from source_alloc.utils import *


# 设置工作目录为当前文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 打印当前运行目录
print(os.getcwd())

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

starting_seed = 0
ending_seed = 50
num_evaluation = ending_seed - starting_seed
DEBUG_FLAG = False
EVAL_CASE = "Time" # ["Performace", "Time"]
if DEBUG_FLAG:
    test_N = [10]
    # test_L = [5]
    selected_method_list = ["prop", "ga"]
else:
    if EVAL_CASE == "Performace":
        test_N = [4, 8, 12, 16, 20]
        L = 5
        selected_method_list = ["greedy_pruned"] # ["prop", "greedy", "no_save", "all_save", "ga"]
        # selected_method_list = ["prop", "greedy", "no_save", "all_save", "ga"]
        # data_name = "resEnhance" # "anomalyDetect" noiseRemove resEnhance
        # data_name_list = ["anomalyDetect", "colorCorrect"] # "anomalyDetect" noiseRemove resEnhance
        data_name_list = ["anomalyDetect", "colorCorrect", "noiseRemove", "resEnhance"] # "anomalyDetect" noiseRemove resEnhance
        num_cores = min(mp.cpu_count() - 1, 10)
    elif EVAL_CASE == "Time":
        test_N = [40]
        L_list = [2]
        # cutoff_L_list = [2,4]
        time_limit = 600
        num_cores = 1
        starting_seed = 0
        ending_seed = num_cores + starting_seed
        num_evaluation = num_cores
        # selected_method_list = ["greedy", "prop", "ga", "greedy_pruned"]
        selected_method_list = ["prop"]
        data_name_list = ["resEnhance"] # "anomalyDetect" noiseRemove resEnhance

class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

@contextlib.contextmanager
def redirect_stdout_to_logger(logger):
    stdout_logger = StreamToLogger(logger)
    stderr_logger = StreamToLogger(logger, log_level=logging.ERROR)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        sys.stdout = stdout_logger
        sys.stderr = stderr_logger
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建文件处理器
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def process_single_seed_testTime(seed, N, L, data_name, selected_method_list, logger):
    with redirect_stdout_to_logger(logger):
        rel_seed = seed - starting_seed
        np.random.seed(seed)

        
        results = {}

        problem = formal_random_problem_reduced(seed, N, L, data_name)
        if N <= 40:
            A, B, b, c, d, B_y, b_y, _, _, _, _ = generate_problem(problem, flag_intact_MIP=True)
            standard_prob_element = {
                "A": A, "B": B, "b": b, "c": c, "d": d,
            }
        else:
            _, _, _, c, d, _, _, _, _, _, _ = generate_problem(problem, flag_intact_MIP=True, flag_too_big_for_A=True)
            standard_prob_element = None
        mproblem_dict = prepare_dict_for_my_solver(problem)

        if "prop" in selected_method_list and N <= 40:
            obj_val, used_level, stored_level, loss_vec, used_time = solve_reduced_problem(problem, standard_prob_element, flag_time=True)
            results["prop"] = {"used_time": used_time}
        else:
            results["prop"] = {"used_time": -1}
        
        if "ga" in selected_method_list and N <= 50:
            start_time = time.time()
            obj_val_ga = solve_full_ga(c, d, mproblem_dict)
            end_time = time.time()
            used_time = end_time - start_time
            results["ga"] = {"used_time": used_time}
        else:
            results["ga"] = {"used_time": -1}
        
        if "greedy" in selected_method_list:
            start_time = time.time()
            obj_val_greedy = solve_greedy(c, d, mproblem_dict)
            end_time = time.time()
            used_time = end_time - start_time
            results["greedy"] = {"used_time": used_time}
        
        if "greedy_pruned" in selected_method_list:
            start_time = time.time()
            obj_val_greedy_pruned = solve_greedy_pruned(c, d, mproblem_dict)
            end_time = time.time()
            used_time = end_time - start_time
            results["greedy_pruned"] = {"used_time": used_time}

        logger.info(f"完成 SEED {seed} 的计算")
        
        return seed, results

def process_single_seed(seed, N, L, data_name, selected_method_list, progress_dict, lock, logger):
    # 重定向该进程的所有输出到logger
    with redirect_stdout_to_logger(logger):
        rel_seed = seed - starting_seed
        np.random.seed(seed)
        
        with lock:
            progress_dict['completed_seeds'] += 1
            total_seeds = ending_seed - starting_seed
            progress = progress_dict['completed_seeds']
            logger.info(f"开始处理 SEED {seed} [{progress}/{total_seeds}]")
        
        problem = formal_random_problem_reduced(seed, N, L, data_name)
        analysis_loss_components(problem)
        
        A, B, b, c, d, B_y, b_y, _, _, _, _ = generate_problem(problem, flag_intact_MIP=True)
        standard_prob_element = {
            "A": A, "B": B, "b": b, "c": c, "d": d,
        }
        mproblem_dict = prepare_dict_for_my_solver(problem)

        results = {}
        
        if "prop" in selected_method_list:
            obj_val, used_level, stored_level, loss_vec = solve_reduced_problem(problem, standard_prob_element)
            results["prop"] = {"obj_val": obj_val, "used_level": used_level, 
                              "stored_level": stored_level, "loss_vec": loss_vec}
        
        if "ga" in selected_method_list:
            obj_val_ga = solve_full_ga(c, d, mproblem_dict)
            results["ga"] = {"obj_val": obj_val_ga[0]}
        
        if "greedy" in selected_method_list:
            obj_val_greedy = solve_greedy(c, d, mproblem_dict)
            # obj_val_greedy = solve_greedy_pruned(c, d, mproblem_dict)
            results["greedy"] = {"obj_val": obj_val_greedy[0]}
        
        if "greedy_pruned" in selected_method_list:
            obj_val_greedy_pruned = solve_greedy_pruned(c, d, mproblem_dict)
            results["greedy_pruned"] = {"obj_val": obj_val_greedy_pruned[0]}
        
        if "no_save" in selected_method_list:
            obj_val_no_save = solve_no_save(c, d, mproblem_dict)
            results["no_save"] = {"obj_val": obj_val_no_save[0]}
        
        if "all_save" in selected_method_list:
            obj_val_all_save = solve_all_save(c, d, mproblem_dict)
            results["all_save"] = {"obj_val": obj_val_all_save[0]}
        
        with lock:
            logger.info(f"完成 SEED {seed} 的计算")
        
        return seed, results

manager = None

def init_manager():
    global manager
    manager = Manager()
    return manager

def main_test_case(log_root_dir):
    # 设置主日志记录器
    logger = setup_logger(os.path.join(log_root_dir, 'logs'))
    logger.info("开始运行主程序")
    
    for data_name in data_name_list:
        for idx_N, N in enumerate(test_N):
            logger.info(f"开始处理数据集 {data_name}, N = {N}")
            
            # 初始化结果矩阵
            results_matrices = {
                "prop": np.zeros((1, num_evaluation)),
                "greedy": np.zeros((1, num_evaluation)),
                "greedy_pruned": np.zeros((1, num_evaluation)),
                "no_save": np.zeros((1, num_evaluation)),
                "all_save": np.zeros((1, num_evaluation)),
                "ga": np.zeros((1, num_evaluation))
            }
            
            # 创建进程池和共享资源
            
            pool = mp.Pool(num_cores)
            manager = Manager()
            progress_dict = manager.dict()
            progress_dict['completed_seeds'] = 0
            lock = manager.Lock()
            
            # 修改process_func，传入logger
            process_func = partial(process_single_seed, 
                                 N=N, 
                                 L=L, 
                                 data_name=data_name, 
                                 selected_method_list=selected_method_list,
                                 progress_dict=progress_dict,
                                 lock=lock,
                                 logger=logger)
            
            # 并行处理所有seed
            for seed, results in pool.imap_unordered(process_func, range(starting_seed, ending_seed)):
                rel_seed = seed - starting_seed
                
                # 更新结果
                for method, result in results.items():
                    if method == "prop":
                        results_matrices[method][0, rel_seed] = result["obj_val"] / N
                    else:
                        results_matrices[method][0, rel_seed] = result["obj_val"] / N
            
            pool.close()
            pool.join()
            
            # 保存结果
            logger.info(f"保存 {data_name}, N = {N} 的结果")
            for method in selected_method_list:
                matlab_save_path = f"{log_root_dir}/{method}_N_{N}_{starting_seed}_{ending_seed}_{data_name}.mat"
                savemat(matlab_save_path, {
                    f"result_summary_{method}": results_matrices[method],
                    "N": N,
                    "starting_seed": starting_seed,
                    "ending_seed": ending_seed,
                    "data_name": data_name
                })
            
            logger.info(f"完成数据集 {data_name}, N = {N} 的处理")


def main_test_case_time(log_root_dir):
    # 设置主日志记录器
    logger = setup_logger(os.path.join(log_root_dir, 'logs'))
    logger.info("开始运行主程序")
    num_L = len(L_list)
    for data_name in data_name_list:
        for idx_N, N in enumerate(test_N):
            
            # 初始化结果矩阵
            results_matrices = {
                "prop": np.zeros((num_L, num_evaluation)),
                "greedy": np.zeros((num_L, num_evaluation)),
                "greedy_pruned": np.zeros((num_L, num_evaluation)),
                "ga": np.zeros((num_L, num_evaluation))
            }
            
            for idx_L, L in enumerate(L_list):
                logger.info(f"开始处理数据集 {data_name}, N = {N}, L = {L}")
                # 创建进程池和共享资源
                
                for seed in range(starting_seed, ending_seed):
                    # 修改process_func，传入logger
                    seed, results = process_single_seed_testTime(seed=seed,
                                        N=N, 
                                        L=L, 
                                        data_name=data_name, 
                                        selected_method_list=selected_method_list,
                                        logger=logger)
                
                    rel_seed = seed - starting_seed
                    
                    # 更新结果
                    for method, result in results.items():
                        results_matrices[method][idx_L, rel_seed] = result["used_time"]
                
            # 保存结果
            logger.info(f"保存 {data_name}, N = {N} 的结果")
            for method in selected_method_list:
                matlab_save_path = f"{log_root_dir}/{method}_N_{N}_L_{L}_{starting_seed}_{ending_seed}_{data_name}.mat"
                savemat(matlab_save_path, {
                    f"time_summary_{method}": results_matrices[method],
                    "N": N,
                    "L": L,
                    "starting_seed": starting_seed,
                    "ending_seed": ending_seed,
                    "data_name": data_name
                })
            
            logger.info(f"完成数据集 {data_name}, N = {N} 的处理")

if __name__ == "__main__":
    if EVAL_CASE == "Performace":
        log_root_dir = "logs_eval_4c3"
    elif EVAL_CASE == "Time":
        log_root_dir = "logs_eval_4c_time"

    if not os.path.exists(log_root_dir):
        os.makedirs(log_root_dir)
    
    try:
        if EVAL_CASE == "Performace":
            main_test_case(log_root_dir)
        elif EVAL_CASE == "Time":
            main_test_case_time(log_root_dir)
        send_to_Gmail(subject=f"In Eval 4c, All Programs successfully finished",
                    body=f"Congratulations! All programs have successfully finished.")
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}", exc_info=True)
        send_to_Gmail(subject=f"Eval 4c Programs Failed",
                  body=f"Unfortunately, all programs have failed. Please check the logs for more details.")
    finally:
        if manager is not None:
            manager.shutdown()
