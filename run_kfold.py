import logging
import os
import subprocess
import sys
import time
from importlib.util import find_spec

import pandas as pd

from config import COMPETITIONS, MODEL_TRACES_DIR, TABLES_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COMPS = list(COMPETITIONS.keys())


def get_python_cmd() -> str:
    """
    Resolve Python launcher in a cross-platform-safe way.
    - Prefer the current interpreter (`sys.executable`) when available.
    - Fall back to `python` on Windows and `python3` elsewhere.
    """
    if sys.executable:
        return sys.executable
    return "python" if os.name == "nt" else "python3"

def check_optimization() -> None:
    """Warn if JAX/NumPyro is unavailable; fall back to PyMC default sampler."""
    if find_spec("numpyro") and find_spec("jax"):
        logger.info("JAX/NumPyro available — MCMC will be hardware-accelerated.")
    else:
        logger.warning("JAX/NumPyro not installed. Sampling will be significantly slower.")
        logger.warning("Install with: pip install -r requirements-accelerated.txt")
        logger.warning("Continuing in 5 seconds...")
        time.sleep(5)

def run_step(step_cmd: list[str], env: dict[str, str]) -> None:
    """Run a subprocess pipeline step; exit the process on non-zero return code."""
    process = subprocess.Popen(step_cmd, env=env, stdout=sys.stdout, stderr=subprocess.STDOUT)
    process.communicate()
    if process.returncode != 0:
        logger.error("Step %s failed with return code %s", ' '.join(step_cmd), process.returncode)
        sys.exit(1)

def main() -> None:
    logger.info("Starting 4-Fold Cross Validation Pipeline")
    check_optimization()
    python_cmd = get_python_cmd()
    logger.info("Using Python executable: %s", python_cmd)

    single_holdout = os.environ.get("PRS_HOLDOUT")
    if single_holdout:
        if single_holdout not in COMPS:
            logger.error("Invalid holdout '%s'. Must be one of %s", single_holdout, COMPS)
            sys.exit(1)
        comps_to_run = [single_holdout]
        logger.info("Running single holdout: %s", single_holdout)
    else:
        comps_to_run = COMPS
        logger.info("Running full cross-validation over all comps: %s", comps_to_run)

    results = []

    for holdout in comps_to_run:
        fold = COMPS.index(holdout)
        logger.info("%s", '=' * 60)
        logger.info("FOLD %d/4: Holdout = %s", fold + 1, holdout)
        logger.info("%s", '=' * 60)

        env = os.environ.copy()
        env["PRS_HOLDOUT"] = holdout

        # Check if trace already exists on the persistent volume to avoid rerunning MCMC (55 min/fold)
        trace_file = MODEL_TRACES_DIR / f"pooled_trace_{holdout}.nc"
        if trace_file.exists() and os.environ.get("FORCE_RETRAIN", "") != "1":
            logger.info("Trace for %s already exists at %s. Skipping MCMC sampling for this fold.", holdout, trace_file)
            env["FORCE_RETRAIN"] = "0"
        else:
            env["FORCE_RETRAIN"] = "1"


        steps = [
            [python_cmd, "-m", "src.features.train_vaep"],
            [python_cmd, "-m", "src.data.builder"],
            [python_cmd, "-m", "src.models.bayesian"],
            [python_cmd, "-m", "src.models.inference"],
            [python_cmd, "-m", "src.models.validation"]
        ]

        start_time = time.time()
        for step in steps:
            logger.info("--> Executing: %s", ' '.join(step))
            run_step(step, env)

        fold_time = time.time() - start_time
        logger.info("Fold %d completed in %.1f minutes.", fold + 1, fold_time / 60)

        metrics_file = TABLES_DIR / f"holdout_metrics_{holdout}.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            pearson = df['pearson'].iloc[0]
            auc = df['auc'].iloc[0]
            results.append({
                "Fold": fold + 1,
                "Holdout": holdout,
                "Pearson": pearson,
                "AUC": auc,
                "Time_Mins": round(fold_time / 60, 1)
            })
            logger.info("Fold %d Results -> Pearson: %.3f, AUC: %.3f", fold + 1, pearson, auc)
        else:
            logger.error("Metrics file not found for fold %d. Validation step may have failed.", fold + 1)
            sys.exit(1)

    if not single_holdout:
        logger.info("%s", '=' * 60)
        logger.info("FINAL CROSS-VALIDATION RESULTS")
        logger.info("%s", '=' * 60)

        res_df = pd.DataFrame(results)
        print("\n" + res_df.to_string(index=False) + "\n")

        mean_pearson = res_df['Pearson'].mean()
        pearson_min = res_df['Pearson'].min()
        pearson_max = res_df['Pearson'].max()

        logger.info("Mean Pearson Correlation: %.3f (Range: %.3f - %.3f)", mean_pearson, pearson_min, pearson_max)

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(TABLES_DIR / "kfold_results.csv", index=False)
        logger.info("Results saved to %s", TABLES_DIR / "kfold_results.csv")

if __name__ == "__main__":
    main()
