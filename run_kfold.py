import logging
import os
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COMPS = ["Euro_2020", "Euro_2024", "World_Cup_2022", "Bundesliga_2024"]


def get_python_cmd():
    """
    Resolve Python launcher in a cross-platform-safe way.
    - Prefer the current interpreter (`sys.executable`) when available.
    - Fall back to `python` on Windows and `python3` elsewhere.
    """
    if sys.executable:
        return sys.executable
    return "python" if os.name == "nt" else "python3"

def check_optimization():
    """Warn if JAX/NumPyro is unavailable; fall back to PyMC default sampler."""
    if find_spec("numpyro") and find_spec("jax"):
        logger.info("JAX/NumPyro available — MCMC will be hardware-accelerated.")
    else:
        logger.warning("JAX/NumPyro not installed. Sampling will be significantly slower.")
        logger.warning("Install with: pip install -r requirements-accelerated.txt")
        logger.warning("Continuing in 5 seconds...")
        time.sleep(5)

def run_step(step_cmd, env):
    """Run a pipeline subprocess; exit on non-zero return code."""
    process = subprocess.Popen(step_cmd, env=env, stdout=sys.stdout, stderr=subprocess.STDOUT)
    process.communicate()
    if process.returncode != 0:
        logger.error(f"Step {' '.join(step_cmd)} failed with return code {process.returncode}")
        sys.exit(1)

def main():
    logger.info("Starting 4-Fold Cross Validation Pipeline")
    check_optimization()
    python_cmd = get_python_cmd()
    logger.info(f"Using Python executable: {python_cmd}")
    
    results = []
    
    for fold, holdout in enumerate(COMPS):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold + 1}/4: Holdout = {holdout}")
        logger.info(f"{'='*60}")
        
        env = os.environ.copy()
        env["PRS_HOLDOUT"] = holdout  # Inject holdout fold
        
        steps = [
            [python_cmd, "-m", "src.data.builder"],
            [python_cmd, "-m", "src.models.bayesian"],
            [python_cmd, "-m", "src.models.inference"],
            [python_cmd, "-m", "src.models.validation"]
        ]
        
        start_time = time.time()
        for step in steps:
            logger.info(f"--> Executing: {' '.join(step)}")
            run_step(step, env)
            
        fold_time = time.time() - start_time
        logger.info(f"Fold {fold + 1} completed in {fold_time / 60:.1f} minutes.")
        
        metrics_file = Path("outputs/tables/holdout_metrics.csv")
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
            logger.info(f"Fold {fold + 1} Results -> Pearson: {pearson:.3f}, AUC: {auc:.3f}")
        else:
            logger.error(f"Metrics file not found for fold {fold + 1}. Validation step may have failed.")
            sys.exit(1)
            
    logger.info(f"{'='*60}")
    logger.info("FINAL CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    
    res_df = pd.DataFrame(results)
    print("\n" + res_df.to_string(index=False) + "\n")
    
    mean_pearson = res_df['Pearson'].mean()
    pearson_min = res_df['Pearson'].min()
    pearson_max = res_df['Pearson'].max()
    
    logger.info(f"Mean Pearson Correlation: {mean_pearson:.3f} (Range: {pearson_min:.3f} - {pearson_max:.3f})")
    
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    res_df.to_csv("outputs/tables/kfold_results.csv", index=False)
    logger.info("Results saved to outputs/tables/kfold_results.csv")

if __name__ == "__main__":
    main()
