import pandas as pd

# Function to process non-numeric columns
def process_non_numeric(group):
    unique_values = group.unique()
    return unique_values[0] if len(unique_values) == 1 else None

def average_by_target(df_bytarget):
    # Separate numeric and non-numeric columns
    numeric_cols = df_bytarget.select_dtypes(include=['number']).columns
    non_numeric_cols = df_bytarget.select_dtypes(exclude=['number']).columns
    # exlude 'target' column
    if non_numeric_cols[0] == 'target':
        non_numeric_cols = non_numeric_cols[1:]

    # Group by 'target' and calculate mean for numeric columns
    numeric_grouped = df_bytarget.groupby('target')[numeric_cols].mean()

    # Process non-numeric columns
    if len(non_numeric_cols) == 0:
        return numeric_grouped.reset_index()
    
    non_numeric_grouped = (
        df_bytarget.groupby('target')[non_numeric_cols]
        .agg(lambda col: col.unique()[0] if len(col.unique()) == 1 else None)
    )

    # Drop columns with None values (indicating mixed values in groups)
    non_numeric_grouped = non_numeric_grouped.dropna(axis=1, how='any')
    # print dropped columns
    print(f"Dropped columns: {set(non_numeric_cols) - set(non_numeric_grouped.columns)}")

    # Combine numeric and processed non-numeric columns
    df_target_mean = pd.concat([numeric_grouped, non_numeric_grouped], axis=1).reset_index()
    return df_target_mean
