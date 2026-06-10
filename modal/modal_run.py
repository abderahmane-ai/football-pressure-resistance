import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import modal

app = modal.App("prs-kfold")
vol = modal.Volume.from_name("prs-data-vol", create_if_missing=True)

# Resolve project root absolute path relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2.2",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "pymc>=5.10",
        "arviz>=0.17",
        "statsbombpy>=1.13",
        "tqdm>=4.66",
        "shapely>=2.0",
        "pyarrow>=15",
        "numpyro",
        "jax[cuda12]",  # CUDA 12 backend for A100
        "h5netcdf",     # Required backend for writing netcdf4 traces via xarray/arviz
    )
    # Exclude data/ and outputs/ from the upload — they are symlinked from
    # the persistent volume and copying them only to delete them is wasteful.
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/pressure_resistance",
        ignore=[".git", ".venv", "__pycache__", "data", "outputs"],
    )
)

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600,
)
def prepare_data(redo_download: bool = False, redo_data: bool = False) -> None:
    """Pre-flight step to sequentially download and/or process raw data safely."""
    print("=== Modal Data Preparation Starting ===")
    
    os.makedirs("/data/data/raw", exist_ok=True)
    os.makedirs("/data/data/processed", exist_ok=True)
    os.makedirs("/data/outputs", exist_ok=True)

    # Mount is read-only; copy to a writable workspace.
    shutil.copytree("/root/pressure_resistance", "/workspace", dirs_exist_ok=True)
    os.chdir("/workspace")

    import sys
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")

    # Symlink data/ and outputs/
    for d in ("data", "outputs"):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.symlink(f"/data/{d}", d)

    # 1. Handle raw data download
    if redo_download:
        print("[FLAG] Wiping raw data for re-download...")
        shutil.rmtree("/data/data/raw", ignore_errors=True)
        os.makedirs("/data/data/raw", exist_ok=True)

    from run_kfold import COMPS
    from src.data.loader import download_all
    for comp in COMPS:
        events_file = Path(f"/data/data/raw/{comp}/events.parquet")
        frames_dir = Path(f"/data/data/raw/{comp}/frames")
        has_frames = False
        if frames_dir.exists():
            for f in frames_dir.glob("*"):
                if f.suffix in (".parquet", ".pkl"):
                    has_frames = True
                    break
        if redo_download or not events_file.exists() or not has_frames:
            print(f"Downloading/Caching raw data for competition: {comp}")
            download_all(comp)
        else:
            print(f"Raw data for {comp} already cached.")

    # 2. Handle processed data generation
    if redo_data:
        print("[FLAG] Wiping processed data files for rebuilding...")
        for comp in COMPS:
            for f in [f"/data/data/processed/all_pressure_dataset_{comp}.parquet",
                      f"/data/data/processed/holdout_pressure_dataset_{comp}.parquet"]:
                try:
                    os.unlink(f)
                except FileNotFoundError:
                    pass

    import config
    from src.data.builder import build_all_datasets, build_holdout_dataset
    for comp in COMPS:
        os.environ["PRS_HOLDOUT"] = comp
        config.CROSS_VALIDATION_HOLDOUT = comp
        print(f"Building/caching processed datasets for holdout: {comp}")
        build_all_datasets(include_holdout=False)
        build_holdout_dataset()

    print("Committing prepared data to volume...")
    vol.commit()
    print("=== Modal Data Preparation Completed ===")


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": vol},
    timeout=86400,
)
def run_fold(holdout: str, redo_training: bool = False) -> None:
    """Run a single fold pipeline on A100. Uses a try...finally block to commit the volume even if training crashes."""
    print(f"=== Modal Fold Execution Starting on A100 for holdout: {holdout} ===")

    # Mount is read-only; copy to a writable workspace.
    shutil.copytree("/root/pressure_resistance", "/workspace", dirs_exist_ok=True)
    os.chdir("/workspace")
    
    import sys
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")

    # Symlink data/ and outputs/
    for d in ("data", "outputs"):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.symlink(f"/data/{d}", d)

    env = os.environ.copy()
    env["PRS_HOLDOUT"] = holdout
    env["JAX_PLATFORMS"] = "cuda"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    env["JAX_ENABLE_X64"] = "1"
    if redo_training:
        env["PRS_FORCE_RETRAIN"] = "1"

    try:
        print(f"Executing: python run_kfold.py (PRS_HOLDOUT={holdout}) ...")
        process = subprocess.Popen(["python", "run_kfold.py"], env=env)
        process.communicate()

        if process.returncode != 0:
            print(f"=== FOLD {holdout} PIPELINE FAILED ===")
            raise RuntimeError(f"Fold {holdout} pipeline failed with exit code {process.returncode}")
        print(f"=== FOLD {holdout} PIPELINE COMPLETED SUCCESSFULLY ===")
    finally:
        print(f"Committing volume changes for fold {holdout}...")
        vol.commit()


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
)
def aggregate_results() -> None:
    """Sync volume changes, aggregate holdout metrics, print summary table, and copy a default leaderboard."""
    print("=== Modal Aggregating Results ===")
    
    # Reload volume to retrieve files committed by parallel run_fold runs
    vol.reload()

    # Symlink or set path
    shutil.copytree("/root/pressure_resistance", "/workspace", dirs_exist_ok=True)
    os.chdir("/workspace")
    
    import sys
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")

    for d in ("data", "outputs"):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.symlink(f"/data/{d}", d)

    import pandas as pd

    from run_kfold import COMPS

    results = []
    successful_folds = []
    
    for fold, holdout in enumerate(COMPS):
        metrics_file = Path(f"outputs/tables/holdout_metrics_{holdout}.csv")
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            pearson = df['pearson'].iloc[0]
            auc = df['auc'].iloc[0]
            results.append({
                "Fold": fold + 1,
                "Holdout": holdout,
                "Pearson": pearson,
                "AUC": auc,
            })
            successful_folds.append(holdout)
            print(f"Fold {fold + 1} ({holdout}): Pearson={pearson:.3f}, AUC={auc:.3f}")
        else:
            print(f"WARNING: Metrics file for fold {fold + 1} ({holdout}) is missing.")

    if not results:
        print("ERROR: No fold metrics found to aggregate.")
        return

    res_df = pd.DataFrame(results)
    print("\n=== FINAL CROSS-VALIDATION RESULTS ===")
    print(res_df.to_string(index=False))
    
    mean_pearson = res_df['Pearson'].mean()
    pearson_min = res_df['Pearson'].min()
    pearson_max = res_df['Pearson'].max()
    print(f"\nMean Pearson Correlation: {mean_pearson:.3f} (Range: {pearson_min:.3f} - {pearson_max:.3f})")

    # Save aggregated kfold results
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    res_df.to_csv("outputs/tables/kfold_results.csv", index=False)
    print("Saved outputs/tables/kfold_results.csv")

    # Concatenate all fold leaderboards into one file with a holdout column
    leaderboard_parts = []
    for holdout in successful_folds:
        src = Path(f"outputs/tables/prs_leaderboard_{holdout}.csv")
        if src.exists():
            df = pd.read_csv(src)
            df.insert(0, "holdout", holdout)
            leaderboard_parts.append(df)
        else:
            print(f"WARNING: Leaderboard for holdout {holdout} not found at {src}")

    if leaderboard_parts:
        combined = pd.concat(leaderboard_parts, ignore_index=True)
        combined.to_csv("outputs/tables/prs_leaderboard.csv", index=False)
        print(f"Combined {len(leaderboard_parts)} fold leaderboards into prs_leaderboard.csv ({len(combined)} rows)")
    else:
        print("WARNING: No leaderboard files found to combine")

    # Commit volume to persist aggregated tables
    print("Committing aggregated results to volume...")
    vol.commit()


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=86400,
)
def run_pipeline(redo_download: bool = False, redo_data: bool = False, redo_training: bool = False) -> None:
    """Orchestrates the entire pipeline from the cloud. Safe to run detached."""
    print("=== Remote Orchestrator Pipeline Starting ===")
    
    print("\n--- STAGE 1: Data Preparation ---")
    prepare_data.local(redo_download, redo_data)

    from run_kfold import COMPS
    print(f"\n--- STAGE 2: Parallel Model Training ({len(COMPS)} Folds concurrently on A100 GPUs) ---")
    list(run_fold.map(
        COMPS,
        [redo_training] * len(COMPS)
    ))

    print("\n--- STAGE 3: Aggregating Results ---")
    aggregate_results.local()
    
    print("\n=== Remote Orchestrator Pipeline Completed ===")


@app.local_entrypoint()
def main(
    redo_download: bool = False,
    redo_data: bool = False,
    redo_training: bool = False,
    detach: bool = False,
) -> None:
    """
    Parallelized pipeline entrypoint. Runs data preparation first,
    then maps the folds in parallel across A100 GPU containers,
    and finally aggregates the results.
    """
    from run_kfold import COMPS
    print(f"🚀 Triggering Parallelized {len(COMPS)}-Fold Cross Validation Pipeline on Modal.")
    print(f"Flags: download={redo_download}, data={redo_data}, train={redo_training}, detach={detach}")

    if detach:
        call = cast(Any, run_pipeline).spawn(redo_download, redo_data, redo_training)
        print(f"\n🚀 Execution spawned in the background!")
        print(f"Function Call ID: {call.object_id}")
        print("View live logs at: https://modal.com/apps")
        print("Download outputs when done:")
        print("  modal volume get prs-data-vol /outputs ./outputs_modal")
    else:
        cast(Any, run_pipeline).remote(redo_download, redo_data, redo_training)
        print("\n✅ Execution finished successfully.")
        print("Download outputs:")
        print("  modal volume get prs-data-vol /outputs ./outputs_modal")
