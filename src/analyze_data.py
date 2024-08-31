import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataframe(df):
    
    # Check if the first column is the index or unnamed, then remove it if necessary
    if df.columns[0] == df.index.name or df.columns[0].startswith('Unnamed'):
        print("\nThe first column is either the index or unnamed, removing it.")
        df.drop(df.columns[0], axis=1, inplace=True)
        if df.columns.size > 0:
            print("\nThe new first column is:", df.columns[0])
        else:
            print("\nNo columns left after removal.")
    else:
        print("\nThe first column is not the index, it is:", df.columns[0])

    # Clean column names by removing spaces
    df.columns = df.columns.str.replace(' ', '_')
    
    # Check for duplicates and remove them
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"\n{num_duplicates} duplicate rows found.")
        response = input("Do you want to remove duplicates? (yes/no): ").lower()
        if response == 'yes':
            df.drop_duplicates(inplace = True)
            print("Duplicates removed.")
        else:
            print("Duplicates not removed.")
    else:
        print("\nNo duplicates found.")

    # Display basic information about the DataFrame
    print("\nDataFrame Info:")
    df.info()

    # Show columns and their data types
    print("\nColumns and their Data Types:")
    print(df.dtypes)

    # Show unique values in each column
    print("\nUnique values in each column:")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

    # Display rows with null values and ask about removing them
    if df.isnull().any().any():
        print("\nRows with null values found:")
        display(df[df.isnull().any(axis=1)])
        if input("Do you want to remove all rows with null values? (yes/no): ").lower() == 'yes':
            df.dropna(inplace = True)
            print("Null values removed.")
    else:
        print("\nNo null values in DataFrame.")

    # Statistics and detection of outliers for each numerical column
    print("\nStatistics for the dataset:")
    df.describe()

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    #Separate categorical and numberical columns
    cat_col = df.dtypes[df.dtypes == 'object']
    num_col = df.dtypes[df.dtypes != 'object']

    for col in list(cat_col.index):
        print(f"--------------------{col.title()}-------------------------")
        total= df[col].value_counts()
        percent = df[col].value_counts() / df.shape[0]
        df_col = pd.concat([total,percent],keys = ['total','percent'],axis = 1)
        print(df_col)
        print('\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

    # Normalize numerical columns and save in a temporary variable
    normalized_df = df.select_dtypes(include=[np.number]).apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    print(f"--------------------Numerical features -------------------------")

    # Visualization of histograms for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        df.hist(column=numerical_cols, figsize=(12, 6), bins=15)
        plt.suptitle('Histograms for Numerical Columns')
        plt.xticks(rotation=90)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Boxplots of all normalized numerical columns
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data = normalized_df)
        plt.xticks(rotation=90)
        plt.title('Boxplots Numerical Columns')
        plt.show()


# Example usage:
# df = pd.read_csv('your_data.csv')
def eda(df):
    from IPython.display import display
    
    # Separate categorical and numeric columns
    cat_col = df.select_dtypes(include=['object', 'category'])
    num_col = df.select_dtypes(include=['number'])
    
    # Display categorical columns
    print(f"--------------------Categorical features -------------------------")
    for col in cat_col.columns:
        print(f"--------------------{col.title()}-------------------------")
        display(df[col].value_counts())
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
    
    # Display numerical columns
    print(f"--------------------Numerical features -------------------------")
    
    for col in num_col.columns:
        print(f"--------------------{col.title()}-------------------------")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Histogram
        axes[0].hist(df[col], bins=30, edgecolor='k')
        axes[0].set_xlabel(col, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        
        # Boxplot
        sns.boxplot(x=col, data=df, ax=axes[1])
        axes[1].set_xlabel(col, fontsize=12)
        
        plt.show()

