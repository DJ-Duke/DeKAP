"""Benchmark harness for knowledge-allocation solvers (performance vs. wall-clock time)."""
import argparse
import contextlib
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Manager
from pathlib import Path

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source_alloc.utils import *


# Full Gurobi MIP (`prop`) and GA scale poorly with N; keep them within this bound by default.
GUROBI_MIP_MAX_N = 20
GA_MAX_N = 20

_VALID_METHODS = frozenset(
    {"prop", "ga", "greedy", "no_save", "all_save"}
)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run allocation benchmark (eval_4c-style). "
        "Gurobi full MIP (`prop`) and GA are capped at N=%d; greedy heuristics are not. "
        "See README." % GUROBI_MIP_MAX_N
    )
    p.add_argument(
        "--eval-case",
        choices=("Performance", "Time"),
        default=os.environ.get("ALLOC_EVAL_CASE", "Time"),
        help="Performance: objective comparison; Time: wall-clock comparison.",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="Output directory for logs and .mat files (default: logs_eval_4c3 or logs_eval_4c_time).",
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="Larger benchmark preset (Time: N=%d; Performance: N grid 4..%d). "
        "Default is a small quick run (N=5, one seed). Or set ALLOC_EXTENDED=1."
        % (GUROBI_MIP_MAX_N, GUROBI_MIP_MAX_N),
    )
    p.add_argument(
        "--methods",
        default=os.environ.get("ALLOC_METHODS", "").strip() or None,
        help="Comma-separated methods: prop,ga,greedy,no_save,all_save. "
        "Overrides preset for this eval-case. Or set ALLOC_METHODS.",
    )
    return p.parse_args()


args = _parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

starting_seed = 0
ending_seed = 50
num_evaluation = ending_seed - starting_seed
DEBUG_FLAG = False
EVAL_CASE = args.eval_case
EXTENDED = args.extended or os.environ.get("ALLOC_EXTENDED", "").lower() in (
    "1",
    "true",
    "yes",
)

if args.methods:
    selected_method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in selected_method_list if m not in _VALID_METHODS]
    if bad:
        raise SystemExit(f"Unknown --methods entries: {bad}. Valid: {sorted(_VALID_METHODS)}")
else:
    selected_method_list = None

if DEBUG_FLAG:
    test_N = [10]
    L = 5
    L_list = [2]
    selected_method_list = ["prop", "ga"]
    data_name_list = ["resEnhance"]
    num_cores = 1
    starting_seed = 0
    ending_seed = 1
    num_evaluation = ending_seed - starting_seed
elif EXTENDED:
    if EVAL_CASE == "Performance":
        test_N = [4, 8, 12, 16, 20]
        L = 5
        if selected_method_list is None:
            selected_method_list = ["greedy"]
        data_name_list = ["anomalyDetect", "colorCorrect", "noiseRemove", "resEnhance"]
        num_cores = min(mp.cpu_count() - 1, 10)
    elif EVAL_CASE == "Time":
        test_N = [GUROBI_MIP_MAX_N]
        L_list = [2]
        num_cores = 1
        starting_seed = 0
        ending_seed = num_cores + starting_seed
        num_evaluation = num_cores
        if selected_method_list is None:
            selected_method_list = ["prop", "ga", "greedy"]
        data_name_list = ["resEnhance"]
else:
    if EVAL_CASE == "Performance":
        test_N = [10]
        L = 5
        if selected_method_list is None:
            selected_method_list = ["prop", "greedy"]
        data_name_list = ["resEnhance"]
        num_cores = 1
        starting_seed = 0
        ending_seed = 1
        num_evaluation = ending_seed - starting_seed
    elif EVAL_CASE == "Time":
        test_N = [10]
        L_list = [2]
        num_cores = 1
        starting_seed = 0
        ending_seed = 1
        num_evaluation = ending_seed - starting_seed
        if selected_method_list is None:
            selected_method_list = ["prop", "greedy"]
        data_name_list = ["resEnhance"]

if selected_method_list is None:
    raise RuntimeError("selected_method_list was not set")


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

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

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    return logger


def process_single_seed_testTime(seed, N, L, data_name, selected_method_list, logger):
    with redirect_stdout_to_logger(logger):
        np.random.seed(seed)

        results = {}

        problem = formal_random_problem_reduced(seed, N, L, data_name)
        standard_prob_element = None
        if "prop" in selected_method_list and N <= GUROBI_MIP_MAX_N:
            A, B, b, c, d, B_y, b_y, _, _, _, _ = generate_problem(problem, flag_intact_MIP=True)
            standard_prob_element = {
                "A": A,
                "B": B,
                "b": b,
                "c": c,
                "d": d,
            }
        else:
            _, _, _, c, d, _, _, _, _, _, _ = generate_problem(problem, flag_too_big_for_A=True)

        mproblem_dict = prepare_dict_for_my_solver(problem)

        if "prop" in selected_method_list:
            if N <= GUROBI_MIP_MAX_N and standard_prob_element is not None:
                obj_val, _, _, _, used_time = solve_reduced_problem(
                    problem, standard_prob_element, flag_time=True
                )
                results["prop"] = {"used_time": used_time, "obj_val": float(obj_val)}
            else:
                results["prop"] = {"used_time": -1.0, "obj_val": np.nan}

        if "ga" in selected_method_list:
            if N <= GA_MAX_N:
                start_time = time.time()
                obj_val_ga = solve_full_ga(c, d, mproblem_dict)
                used_time = time.time() - start_time
                results["ga"] = {"used_time": used_time, "obj_val": float(obj_val_ga[0])}
            else:
                results["ga"] = {"used_time": -1.0, "obj_val": np.nan}

        if "greedy" in selected_method_list:
            start_time = time.time()
            obj_val_greedy = solve_greedy(c, d, mproblem_dict)
            results["greedy"] = {
                "used_time": time.time() - start_time,
                "obj_val": float(obj_val_greedy[0]),
            }

        if "prop" in results and "greedy" in results:
            pt = results["prop"].get("used_time", np.nan)
            gt = results["greedy"].get("used_time", np.nan)
            po = results["prop"].get("obj_val", np.nan)
            go = results["greedy"].get("obj_val", np.nan)
            logger.info(
                "Compare prop vs greedy (seed=%s, N=%s): time prop=%.6fs greedy=%.6fs; "
                "objective prop=%.8f greedy=%.8f (per-agent prop=%.8f greedy=%.8f)",
                seed,
                N,
                pt,
                gt,
                po,
                go,
                po / N if N and not np.isnan(po) else np.nan,
                go / N if N and not np.isnan(go) else np.nan,
            )

        logger.info("Finished seed %s", seed)

        return seed, results


def process_single_seed(seed, N, L, data_name, selected_method_list, progress_dict, lock, logger):
    with redirect_stdout_to_logger(logger):
        np.random.seed(seed)

        with lock:
            progress_dict["completed_seeds"] += 1
            total_seeds = ending_seed - starting_seed
            progress = progress_dict["completed_seeds"]
            logger.info("Starting seed %s [%s/%s]", seed, progress, total_seeds)

        problem = formal_random_problem_reduced(seed, N, L, data_name)
        analysis_loss_components(problem)

        standard_prob_element = None
        if "prop" in selected_method_list and N <= GUROBI_MIP_MAX_N:
            A, B, b, c, d, B_y, b_y, _, _, _, _ = generate_problem(problem, flag_intact_MIP=True)
            standard_prob_element = {
                "A": A,
                "B": B,
                "b": b,
                "c": c,
                "d": d,
            }
        else:
            _, _, _, c, d, _, _, _, _, _, _ = generate_problem(problem, flag_too_big_for_A=True)

        mproblem_dict = prepare_dict_for_my_solver(problem)

        results = {}

        if "prop" in selected_method_list:
            if N <= GUROBI_MIP_MAX_N and standard_prob_element is not None:
                obj_val, used_level, stored_level, loss_vec = solve_reduced_problem(
                    problem, standard_prob_element
                )
                results["prop"] = {
                    "obj_val": obj_val,
                    "used_level": used_level,
                    "stored_level": stored_level,
                    "loss_vec": loss_vec,
                }
            else:
                results["prop"] = {
                    "obj_val": -1.0,
                    "used_level": None,
                    "stored_level": None,
                    "loss_vec": None,
                }

        if "ga" in selected_method_list:
            if N <= GA_MAX_N:
                obj_val_ga = solve_full_ga(c, d, mproblem_dict)
                results["ga"] = {"obj_val": obj_val_ga[0]}
            else:
                results["ga"] = {"obj_val": -1.0}

        if "greedy" in selected_method_list:
            obj_val_greedy = solve_greedy(c, d, mproblem_dict)
            results["greedy"] = {"obj_val": obj_val_greedy[0]}

        if "no_save" in selected_method_list:
            obj_val_no_save = solve_no_save(c, d, mproblem_dict)
            results["no_save"] = {"obj_val": obj_val_no_save[0]}

        if "all_save" in selected_method_list:
            obj_val_all_save = solve_all_save(c, d, mproblem_dict)
            results["all_save"] = {"obj_val": obj_val_all_save[0]}

        with lock:
            logger.info("Finished seed %s", seed)

        return seed, results


manager = None


def init_manager():
    global manager
    manager = Manager()
    return manager


def main_test_case(log_root_dir):
    logger = setup_logger(os.path.join(log_root_dir, "logs"))
    logger.info("Starting benchmark (Performance)")

    for data_name in data_name_list:
        for idx_N, N in enumerate(test_N):
            logger.info("Dataset %s, N=%s", data_name, N)

            results_matrices = {m: np.zeros((1, num_evaluation)) for m in selected_method_list}

            pool = mp.Pool(num_cores)
            mgr = Manager()
            progress_dict = mgr.dict()
            progress_dict["completed_seeds"] = 0
            lock = mgr.Lock()

            process_func = partial(
                process_single_seed,
                N=N,
                L=L,
                data_name=data_name,
                selected_method_list=selected_method_list,
                progress_dict=progress_dict,
                lock=lock,
                logger=logger,
            )

            for seed, results in pool.imap_unordered(process_func, range(starting_seed, ending_seed)):
                rel_seed = seed - starting_seed

                for method, result in results.items():
                    results_matrices[method][0, rel_seed] = result["obj_val"] / N

            pool.close()
            pool.join()

            logger.info("Saving results for %s, N=%s", data_name, N)
            for method in selected_method_list:
                matlab_save_path = (
                    f"{log_root_dir}/{method}_N_{N}_{starting_seed}_{ending_seed}_{data_name}.mat"
                )
                savemat(
                    matlab_save_path,
                    {
                        f"result_summary_{method}": results_matrices[method],
                        "N": N,
                        "starting_seed": starting_seed,
                        "ending_seed": ending_seed,
                        "data_name": data_name,
                    },
                )

            logger.info("Done dataset %s, N=%s", data_name, N)


def main_test_case_time(log_root_dir):
    logger = setup_logger(os.path.join(log_root_dir, "logs"))
    logger.info("Starting benchmark (Time)")
    num_L = len(L_list)
    for data_name in data_name_list:
        for idx_N, N in enumerate(test_N):

            results_matrices = {m: np.zeros((num_L, num_evaluation)) for m in selected_method_list}
            obj_matrices = {m: np.full((num_L, num_evaluation), np.nan) for m in selected_method_list}

            for idx_L, L in enumerate(L_list):
                logger.info("Dataset %s, N=%s, L=%s", data_name, N, L)

                for seed in range(starting_seed, ending_seed):
                    seed, results = process_single_seed_testTime(
                        seed=seed,
                        N=N,
                        L=L,
                        data_name=data_name,
                        selected_method_list=selected_method_list,
                        logger=logger,
                    )

                    rel_seed = seed - starting_seed

                    for method, result in results.items():
                        results_matrices[method][idx_L, rel_seed] = result["used_time"]
                        if "obj_val" in result:
                            obj_matrices[method][idx_L, rel_seed] = result["obj_val"]

            logger.info("Saving timing and objective results for %s, N=%s", data_name, N)
            for method in selected_method_list:
                matlab_save_path = (
                    f"{log_root_dir}/{method}_N_{N}_L_{L}_{starting_seed}_{ending_seed}_{data_name}.mat"
                )
                savemat(
                    matlab_save_path,
                    {
                        f"time_summary_{method}": results_matrices[method],
                        f"objective_summary_{method}": obj_matrices[method],
                        "N": N,
                        "L": L,
                        "starting_seed": starting_seed,
                        "ending_seed": ending_seed,
                        "data_name": data_name,
                    },
                )

            logger.info("Done dataset %s, N=%s", data_name, N)


if __name__ == "__main__":
    if args.log_dir:
        log_root_dir = args.log_dir
    elif EVAL_CASE == "Performance":
        log_root_dir = "logs_eval_4c3"
    else:
        log_root_dir = "logs_eval_4c_time"

    os.makedirs(log_root_dir, exist_ok=True)

    try:
        if EVAL_CASE == "Performance":
            main_test_case(log_root_dir)
        elif EVAL_CASE == "Time":
            main_test_case_time(log_root_dir)
        logging.info("Benchmark finished successfully.")
    except Exception as e:
        logging.error("Benchmark failed: %s", str(e), exc_info=True)
        raise
    finally:
        if manager is not None:
            manager.shutdown()
