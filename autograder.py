import os
import sys
import csv
import time
import glob
import math
import logging
import datetime
import subprocess

import numpy as np
import pandas as pd
import argparse
import json
import tempfile

from typing import Optional

LOG_FILE = "autograder.log"
RESULT_CSV = "results.csv"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _load_or_init_payload(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    # Ensure tests exists if we’re going to append to it
    if "tests" not in data or not isinstance(data["tests"], list):
        data["tests"] = []
    return data


RESULT_JSON_NAME = "results.json"
RESULTS_DIR = "/Users/vamsi/Downloads/Autograder 2"

def _write_json(path: str, payload: dict) -> None:
    """
    Write the given payload as JSON to the given path, creating directories if needed.
    Overwrites the file if it already exists.
    """
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def write_test_to_json(
    duration: Optional[float] = None,
    epsilon_max: Optional[float] = None,
    rho: Optional[float] = None,
    max_score: Optional[int] = 0,
    status: Optional[str] = None,
    filename: str = RESULT_JSON_NAME,
) -> str:
    """
    Appends a single test result to target_dir/results.json.
    Only includes provided fields. Packs run params under tests[i].extra_data.
    """
    path = os.path.join(RESULTS_DIR, filename)
    payload = _load_or_init_payload(path)

    test_entry = {}

    name = f"Testcase: {duration=}, ε={epsilon_max}, 𝝆={rho}"

    test_entry["name"] = name

    if max_score is not None:
        test_entry["max_score"] = max_score

    if status is not None:
        test_entry["status"] = status
        if status == "passed":
            test_entry["score"] = max_score
        else:
            test_entry["score"] = 0

    # Collect run params as extra_data only if present
    extra = {}
    if duration is not None:
        extra["duration"] = duration
    if epsilon_max is not None:
        extra["epsilon_max"] = epsilon_max
    if rho is not None:
        extra["rho"] = rho
    if extra:
        test_entry["extra_data"] = extra

    # If *nothing* provided, still append a minimal object to record the run
    if not test_entry:
        test_entry = {}

    payload["tests"].append(test_entry)
    _write_json(path, payload)
    return path



def find_latest_csv():
    files = glob.glob("*.csv")
    if not files:
        return None

    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def write_to_csv(target_dir, duration, epsilon_max, rho, status):
    file_exists = os.path.exists(RESULT_CSV)

    with open(RESULT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["target_dir", "duration", "epsilon_max", "rho", "status"])

        # Write header only if file does not exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {"target_dir": target_dir, "duration": duration, "epsilon_max": epsilon_max, "rho": rho, "status": status}
        )

def run_and_grade(target_dir, d, epsilon_max, rho):
    origin_dir = os.getcwd()
    if not os.path.isdir(target_dir):
        logging.error(f"[FAIL] Target folder '{target_dir}' does not exist")
        return False, "FAIL-DIR_TARGET_NOT_FOUND"

    # Change directory
    os.chdir(target_dir)
    logging.info(f"[OK] Changed directory to '{target_dir}'")

    try:
        # Run the program
        cmd = ["make", "run_project", f"d={d}", f"epsilon_max={epsilon_max}", f"rho={rho}"]
        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=d * 1.2)
        except subprocess.TimeoutExpired:
            logging.error(f"[FAIL] Process exceeded timeout of {d * 1.2} seconds and was killed")
            return False, "FAIL-SUBPROCESS_TIMEOUT"
        except Exception as e:
            logging.error(f"[FAIL] Subprocess failed: {e}")
            return False, "FAIL-SUBPROCESS_FAILURE"
        end = time.time()

        if result.returncode != 0:
            logging.error(f"[FAIL] Process exited with code {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
            return False, "FAIL-SUBPROCESS_EXITED"

        # Check runtime
        elapsed = end - start
        if elapsed - d < 0.0:
            logging.error(f"[FAIL] Runtime was {elapsed:.2f}s, expected ~{d}s")
            return False, "FAIL-LOGIC_DURATION_TIMEOUT"
        logging.info(f"[OK] Runtime duration ~{d}s")

        # Find the CSV file
        csv_file = find_latest_csv()
        if not csv_file:
            logging.error("[FAIL] No CSV file found in current directory")
            return False, "FAIL-READCSV_NOTFOUND"
        logging.info(f"[OK] Found CSV file: {csv_file}")

        # Read CSV
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logging.error("[FAIL] Could not read CSV:", e)
            return False, "FAIL-READCSV_FAILURE"

        # Check columns
        if not {"actual_time", "local_time"}.issubset(df.columns):
            logging.error("[FAIL] CSV missing required columns")
            return False, "FAIL-READCSV_MISSING_COLS"

        actual = df["actual_time"].tolist()
        local = df["local_time"].tolist()

        # Check actual_time increments ~1s
        for i in range(1, len(actual)):
            dt = actual[i] - actual[i-1]
            if not (0.95 <= dt <= 1.05):
                logging.error(f"[FAIL] actual_time not incrementing properly at index {i}, got {dt:.3f}")
                return False, "FAIL-LOGIC_TIME_INCREMENT_ERROR"
        logging.info("[OK] actual_time increments are ~1s apart")

        # Check drift factor rho
        x = np.array(actual)
        y = np.array(local)
        slope = (x @ y) / (x @ x)
        if not math.isclose(slope, 1 + rho, rel_tol=0.05):
            logging.error(f"[FAIL] Drift slope mismatch: got {slope:.5f}, expected {1+rho:.5f}")
            return False, "FAIL-LOGIC_DRIFT_MISMATCH"
        logging.info("[OK] local_time drifting by ~rho")

        # Check absolute difference within epsilon_max bounds
        diffs = np.abs(x - y)
        max_bound = 1.05 * epsilon_max
        if not all(d <= max_bound for d in diffs):
            logging.error(f"[FAIL] Differences not within [0, {max_bound:.3f}], diffs: {diffs}")
            return False, "FAIL-LOGIC_ERROR_CONSTRAINT_NOT_SATISFIED"
        logging.info("[OK] Differences within epsilon_max bounds")

        logging.info("[PASS] All checks passed ✅")
        return True, "PASS"
    finally:
        os.chdir(origin_dir)
        logging.info(f"[OK] Returned to main directory '{origin_dir}'")

def run_tests_for_target(target_dir, testcase_list):
    for testcase in testcase_list:
        duration, epsilon_max, rho = testcase
        
        print(f"Running Testcase: {duration=}, ε={epsilon_max}, 𝝆={rho}")
        _, status = run_and_grade(target_dir, *testcase)
        if status == "PASS":
            print("Testcase passed")
        else:
            print("Testcase failed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, default=None, help="Directory for the student submission")

    args = parser.parse_args()
    # List of tuples (d, epsilon_max, rho) for each testcase
    testcase_list = [
        (15, 0.02, 1e-6),
        (20, 0.01, 0.0),
        (25, 0.02, 0.01),
        (30, 0.2, 1e-6),
        (40, 0.2, 0.01),
    ]


    run_tests_for_target(args.submission, testcase_list)

