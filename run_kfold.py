import os
import subprocess
import sys
import pandas as pd
from pathlib import Path
import time
import logging

# Professional logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COMPS = ["Euro_2020", "Euro_2024", "World_Cup_2022", "Bundesliga_2024"]

def check_optimization():
    """Checks if hardware-accelerated sampling is available and advises the user."""
    try:
        import numpyro
        import jax
        logger.info("Optimization: JAX/NumPyro is INSTALLED. MCMC sampling will be hardware-accelerated.")
    except ImportError:
        logger.warning("Optimization: JAX/NumPyro is NOT installed.")
        logger.warning("For professional, optimized performance (minutes instead of hours), please run:")
        logger.warning("    pip install numpyro jax jaxlib")
        logger.warning("The script will proceed using the default PyMC Python sampler (which is significantly slower).")
        logger.warning("Sleeping for 5 seconds before starting...")
        time.sleep(5)

def run_step(step_cmd, env):
    """Executes a pipeline step and streams output natively."""
    process = subprocess.Popen(step_cmd, env=env, stdout=sys.stdout, stderr=subprocess.STDOUT)
    process.communicate()
    if process.returncode != 0:
        logger.error(f"Step {' '.join(step_cmd)} failed with return code {process.returncode}")
        sys.exit(1)

def main():
    logger.info("Starting 4-Fold Cross Validation Pipeline")
    check_optimization()
    
    results = []
    
    for fold, holdout in enumerate(COMPS):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold + 1}/4: Holdout = {holdout}")
        logger.info(f"{'='*60}")
        
        # Dynamically inject the holdout dataset into the environment
        env = os.environ.copy()
        env["PRS_HOLDOUT"] = holdout
        
        steps = [
            ["python3", "-m", "src.data.builder"],
            ["python3", "-m", "src.models.bayesian"],
            ["python3", "-m", "src.models.inference"],
            ["python3", "-m", "src.models.validation"]
        ]
        
        start_time = time.time()
        for step in steps:
            logger.info(f"--> Executing: {' '.join(step)}")
            run_step(step, env)
            
        fold_time = time.time() - start_time
        logger.info(f"Fold {fold + 1} completed in {fold_time / 60:.1f} minutes.")
        
        # Extract and record metrics
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
    
    # Save aggregate results
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    res_df.to_csv("outputs/tables/kfold_results.csv", index=False)
    logger.info("Comprehensive results saved to outputs/tables/kfold_results.csv")

if __name__ == "__main__":
    main()
