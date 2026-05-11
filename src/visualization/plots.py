import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import TABLES_DIR, FIGURES_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    sns.set_theme(style="whitegrid")
except AttributeError:
    sns.set_style("whitegrid")

def plot_prs_leaderboard():
    """Plot Top 20 PRS leaderboard with role and scenario info."""
    path = TABLES_DIR / "prs_leaderboard.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path).head(20)
    plt.figure(figsize=(12, 10))
    
    # Plot error bars
    plt.errorbar(x=df['mean_PRS'], y=df['player_name'], 
                 xerr=[df['mean_PRS'] - df['hdi_5%'], df['hdi_95%'] - df['mean_PRS']],
                 fmt='o', color='royalblue', ecolor='lightsteelblue', capsize=3, label='PRS (mean ± 90% HDI)')
    
    # Add scenario labels
    for i, (_, row) in enumerate(df.iterrows()):
        plt.text(row['hdi_95%'] + 0.002, i, f"[{row['position_group'][0]} | {row['best_under_scenario']}]", 
                 va='center', fontsize=9, color='grey')
        
    plt.axvline(0, color='crimson', linestyle='--', alpha=0.6)
    plt.title('Pressure Resistance Score: Expected Value Gain per 360 Dual\nTop 20 Players (Min. 20 Duals)', fontsize=14)
    plt.xlabel('PRS (Expected xT Gain above Population Baseline)', fontsize=12)
    plt.ylabel('Player', fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "1_leaderboard.png", dpi=300)
    plt.close()
    logger.info("Saved Leaderboard plot.")

def plot_feature_importance():
    """Plot standardized coefficients (Beta) for spatial features."""
    path = TABLES_DIR / "feature_importance.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path, index_col=0).sort_values(by='mean', ascending=True)
    plt.figure(figsize=(10, 8))
    
    colors = ['crimson' if x < 0 else 'forestgreen' for x in df['mean']]
    plt.barh(df.index, df['mean'], xerr=[df['mean'] - df['hdi_5%'], df['hdi_95%'] - df['mean']], 
             color=colors, alpha=0.7, ecolor='grey')
    
    plt.axvline(0, color='black', lw=1)
    plt.title('Which Spatial Factors Degrade Composure?\nBayesian Standardized Coefficients (Fixed Effects)', fontsize=13)
    plt.xlabel('Effect on Value Preserved (Standardized Beta)', fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_feature_importance.png", dpi=300)
    plt.close()
    logger.info("Saved Feature Importance plot.")

def plot_stability_analysis():
    """Scatter plot: Training PRS vs Holdout Residuals."""
    path = TABLES_DIR / "holdout_correlation_data.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path)
    plt.figure(figsize=(9, 8))
    
    sns.regplot(data=df, x='mean_PRS', y='residual', 
                scatter_kws={'alpha':0.5, 'color':'royalblue'}, 
                line_kws={'color':'crimson', 'ls':'--'})
    
    # Label top 5 training players
    top_players = df.sort_values(by='mean_PRS', ascending=False).head(5)
    for _, row in top_players.iterrows():
        plt.text(row['mean_PRS'] + 0.001, row['residual'], row['player_name_train'], fontsize=9)
        
    plt.axhline(0, color='grey', ls=':', alpha=0.5)
    plt.axvline(0, color='grey', ls=':', alpha=0.5)
    
    from config import CROSS_VALIDATION_HOLDOUT
    holdout_name = CROSS_VALIDATION_HOLDOUT.replace("_", " ")
    plt.title(f'Cross-Tournament Stability Proof\nTraining PRS vs {holdout_name} Holdout Residuals', fontsize=13)
    plt.xlabel('Training PRS (Estimated Composure Bonus)', fontsize=11)
    plt.ylabel('Holdout Residual (Actual Value Gain above Predicted)', fontsize=11)
    
    # Annotation
    corr_path = TABLES_DIR / "holdout_metrics.csv"
    if corr_path.exists():
        metrics = pd.read_csv(corr_path).iloc[0]
        plt.text(df['mean_PRS'].min(), df['residual'].max(), 
                 f"Pearson r = {metrics['pearson']:.3f}\n(p < 0.01)", 
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "7_stability_scatter.png", dpi=300)
    plt.close()
    logger.info("Saved Stability Scatter plot.")

def plot_marginal_curves():
    """Plot population-level success curves for key features."""
    dist_path = TABLES_DIR / "marginal_dist.csv"
    arc_path = TABLES_DIR / "marginal_arc.csv"
    
    if dist_path.exists():
        df = pd.read_csv(dist_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df['value'], df['mean_p'], color='blue', lw=2)
        plt.fill_between(df['value'], df['hdi_5%'], df['hdi_95%'], color='blue', alpha=0.15)
        plt.title('Population Expected Value vs Nearest Opponent Distance', fontsize=12)
        plt.xlabel('Distance (Yards)')
        plt.ylabel('E[xT Preserved]')
        plt.savefig(FIGURES_DIR / "3_marginal_dist.png")
        plt.close()
        
    if arc_path.exists():
        df = pd.read_csv(arc_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df['value'], df['mean_p'], color='purple', lw=2)
        plt.fill_between(df['value'], df['hdi_5%'], df['hdi_95%'], color='purple', alpha=0.15)
        plt.title('Population Expected Value vs Opponent Coverage Arc', fontsize=12)
        plt.xlabel('Angular Span of Opponents (Radians)')
        plt.ylabel('E[xT Preserved]')
        plt.savefig(FIGURES_DIR / "3_marginal_arc.png")
        plt.close()

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_prs_leaderboard()
    plot_feature_importance()
    plot_marginal_curves()
    plot_stability_analysis()
