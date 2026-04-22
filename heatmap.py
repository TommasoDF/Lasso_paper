import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # Define primary paths based on your project structure
    data_path = Path("Data/clean_data/final_macro_topic_features.csv")
    output_path = Path("Data/clean_data/corr_topics_only_heatmap.png")

    # Fallback: Check the current working directory if the structured path doesn't exist
    if not data_path.exists():
        data_path = Path("final_macro_topic_features.csv")
        output_path = Path("corr_topics_only_heatmap.png")

    print(f"Reading data from: {data_path}")
    try:
        # Read the processed features data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find '{data_path}'. Please ensure the file is in the correct location.")
        return

    # The 10 macro indicators defined in your extraction script
    macro_indicators = [
        'Fed Funds Effective Rate',
        '10-Year Treasury Yield',
        'VIX Volatility Index',
        'Financial Stress Index',
        'High Yield Option-Adjusted Spread',
        'Trade Weighted USD Index',
        '3-Month Treasury Yield',
        'WTI Crude Oil',
        '10-Year Breakeven Inflation Rate',
        'USD to JPY'
    ]

    print("Excluding macro indicators...")
    # Drop the macro columns (errors='ignore' ensures it won't crash if one is missing)
    df_topics = df.drop(columns=macro_indicators, errors='ignore')
    
    print(f"Calculating correlation matrix for the remaining {df_topics.shape[1]} news topics...")
    corr_matrix = df_topics.corr()

    print("Generating heatmap...")
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    sns.heatmap(corr_matrix, 
                cmap='coolwarm', 
                center=0, 
                xticklabels=False, # Hiding labels as there are 180 features
                yticklabels=False, 
                cbar_kws={'label': 'Correlation coefficient'})
    
    plt.title('Correlation matrix of news-topic series', pad=20, fontsize=14)
    plt.tight_layout()
    
    # Create the output directory if it doesn't exist and save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Heatmap saved to: {output_path}")

if __name__ == "__main__":
    main()