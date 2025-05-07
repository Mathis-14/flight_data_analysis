import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Download the dataset
url = "https://www.dropbox.com/scl/fi/fn2v4vyrblvzbwe2r3i13/Flight.csv?rlkey=w3uzjl5m1wpa5hf1sgdcl73y4&dl=1"
response = requests.get(url)
data = StringIO(response.text)
df = pd.read_csv(data)

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Distribution of numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
for i, col in enumerate(numerical_cols[:4], 1):  # Plot first 4 numerical columns
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Box plots for numerical columns
plt.figure(figsize=(15, 6))
df.boxplot()
plt.title('Box Plots of Numerical Variables')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()

# Additional analysis for categorical columns if they exist
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\nCategorical Columns Analysis:")
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())
        
        # Create bar plots for categorical variables
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()

print("\nEDA completed! Check the generated plots for visual insights.")
print("ok")
