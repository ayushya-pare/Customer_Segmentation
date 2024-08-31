import seaborn as sns
import matplotlib.pyplot as plt

def plot_segment_distribution(df, segment_col, feature_col):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=feature_col, hue=segment_col, palette='viridis')
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Customers')
    plt.title(f'Number of Customers in Each Segment by {feature_col}')
    plt.legend(title=feature_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_mean_charges(df, group_by_col, mean_col):
    mean_total_charges = df.groupby(group_by_col)[mean_col].mean().reset_index()
    plt.figure(figsize=(14, 8))
    sns.barplot(data=mean_total_charges, x=group_by_col[1], y=mean_col, hue=group_by_col[0], palette='viridis')
    plt.xlabel('Customer Segment')
    plt.ylabel(f'Mean {mean_col}')
    plt.title(f'Mean {mean_col} by {group_by_col[1]} for Each Segment')
    plt.legend(title=group_by_col[1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
