
import numpy as np
import pandas as pd

def find_outliers_iqr(df): # replace iqr outliers with the median value

    columns = df.select_dtypes(include="number").columns # select all numerical columns

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        outliers = ((df[col] > upper_bound) | (df[col] < lower_bound))
        median_val = df[col].median()

        df.loc[outliers, col] = median_val

    return df

def find_outliers_zScore(df, threshold=3.0):

    columns = df.select_dtypes(include="number").columns  # select all numerical columns
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        median = df[col].median()

        z_score = (df[col] - mean) / std

        outliers = abs(z_score) > threshold
        df.loc[outliers, col] = median

    return df



# Creating test data

# Set random seed for reproducibility
np.random.seed(42)

# Generate a test DataFrame with 100 rows and 10 numerical columns
data = {
    f'col{i}': np.random.normal(loc=50, scale=10, size=100) for i in range(1, 11)
}
df_test = pd.DataFrame(data)

# Inject some outliers
df_test.loc[5, 'col1'] = 200   # extreme high
df_test.loc[10, 'col2'] = -100  # extreme low
df_test.loc[20, 'col3'] = 180
df_test.loc[30, 'col4'] = -90
df_test.loc[40, 'col5'] = 170

df_cleaned = find_outliers_iqr(df_test)

df_cleaned_z = find_outliers_zScore(df_test, threshold=3.0)


print(df_test.head(11))
print(df_cleaned.head(11))
print(df_cleaned_z.head(11))
