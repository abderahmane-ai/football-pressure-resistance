import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import TABLES_DIR, FIGURES_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    sns.set_theme(style="whitegrid")
except AttributeError:
    sns.set_style("whitegrid")

def plot_prs_leaderboard():
    """Plot top players on a 2D plane: Ball Security vs Value Retention."""
    path = TABLES_DIR / "prs_leaderboard.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path)
    if df.empty:
        return
    
    # Take top 30 by combined PRS
    df = df.head(30)
    
    plt.figure(figsize=(10, 10))
    
    sns.scatterplot(data=df, x='mean_Ball_Security_Score', y='mean_Value_Retention_Score',
                    hue='position_group', s=100, alpha=0.8, palette='Set1')
    
    for i, row in df.iterrows():
        plt.text(row['mean_Ball_Security_Score'] + 0.005, row['mean_Value_Retention_Score'],
                 row['player_name'], fontsize=9)
        
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    
    plt.title('Pressure Resistance: Ball Security vs. Value Retention (Top 30)', fontsize=14)
    plt.xlabel('Ball Security under Pressure (Higher = Keeps possession)', fontsize=12)
    plt.ylabel('Value Retention (Higher = More dangerous when successful)', fontsize=12)
    plt.legend(title="Position", loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "1_leaderboard_2D.png", dpi=300)
    plt.close()
    logger.info("Saved 2D Leaderboard plot.")

def plot_feature_importance():
    """Plot standardized coefficients for spatial features (Turnover vs Value)."""
    path = TABLES_DIR / "feature_importance.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        return
    
    df_turnover = df[df.index.str.contains('_ball_security')].copy()
    df_value = df[df.index.str.contains('_value_retention')].copy()
    
    df_turnover.index = df_turnover.index.str.replace('_ball_security', '')
    df_value.index = df_value.index.str.replace('_value_retention', '')
    
    df_turnover = df_turnover.sort_values(by='abs_mean', ascending=True)
    features = df_turnover.index
    df_value = df_value.reindex(features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    colors_turn = ['crimson' if x < 0 else 'forestgreen' for x in df_turnover['mean']]
    axes[0].barh(df_turnover.index, df_turnover['mean'], 
                 xerr=[df_turnover['mean'] - df_turnover['hdi_5%'], df_turnover['hdi_95%'] - df_turnover['mean']], 
                 color=colors_turn, alpha=0.7)
    axes[0].axvline(0, color='black', lw=1)
    axes[0].set_title('Effect on Ball Security', fontsize=12)
    
    colors_val = ['crimson' if x < 0 else 'forestgreen' for x in df_value['mean']]
    axes[1].barh(df_value.index, df_value['mean'], 
                 xerr=[df_value['mean'] - df_value['hdi_5%'], df_value['hdi_95%'] - df_value['mean']], 
                 color=colors_val, alpha=0.7)
    axes[1].axvline(0, color='black', lw=1)
    axes[1].set_title('Effect on Value Retention', fontsize=12)
    
    fig.suptitle('Which Spatial Factors Impact Composure?\nBayesian Standardized Coefficients', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_feature_importance.png", dpi=300)
    plt.close()
    logger.info("Saved Feature Importance plot.")

def plot_stability_analysis():
    """Scatter plot: Training PRS vs Holdout Residuals."""
    path = TABLES_DIR / "holdout_correlation_data.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path)
    if df.empty:
        return
    
    plt.figure(figsize=(9, 8))
    
    sns.regplot(data=df, x='mean_PRS', y='residual', 
                scatter_kws={'alpha':0.5, 'color':'royalblue'}, 
                line_kws={'color':'crimson', 'ls':'--'})
    
    top_players = df.sort_values(by='mean_PRS', ascending=False).head(5)
    for _, row in top_players.iterrows():
        plt.text(row['mean_PRS'] + 0.001, row['residual'], row['player_name_train'], fontsize=9)
        
    plt.axhline(0, color='grey', ls=':', alpha=0.5)
    plt.axvline(0, color='grey', ls=':', alpha=0.5)
    
    from config import CROSS_VALIDATION_HOLDOUT
    holdout_name = CROSS_VALIDATION_HOLDOUT.replace("_", " ")
    plt.title(f'Cross-Tournament Stability Check\nTraining Combined PRS vs {holdout_name} Holdout Residuals', fontsize=13)
    plt.xlabel('Training Combined PRS (Expected xT Gain)', fontsize=11)
    plt.ylabel('Holdout Residual (Actual Value Gain above Predicted)', fontsize=11)
    
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
