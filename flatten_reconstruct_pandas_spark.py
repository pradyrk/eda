
# Flatten and Reconstruct Pandas DataFrames with Spark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import col

# Utility functions

def flatten_pandas_dataframes(df_list):
    flat_data = []
    
    for df_id, df in enumerate(df_list):
        df = df.copy()
        
        # Support MultiIndex for index and columns
        df.columns = pd.MultiIndex.from_tuples(
            [(col,) if not isinstance(col, tuple) else col for col in df.columns]
        )
        df.index = pd.MultiIndex.from_tuples(
            [(idx,) if not isinstance(idx, tuple) else idx for idx in df.index]
        )

        index_names = df.index.names
        column_names = df.columns.names

        for row_number, (row_index, row) in enumerate(df.iterrows()):
            for col_number, (col_name, value) in enumerate(row.items()):
                flat_data.append({
                    'df_id': df_id,
                    'row_number': row_number,
                    'col_number': col_number,
                    'row_index': row_index,
                    'col_name': col_name,
                    'index_names': index_names,
                    'column_names': column_names,
                    'value': value
                })

    return flat_data

def create_spark_dataframe_from_flat(spark: SparkSession, flat_data):
    def serialize_row(row):
        return Row(
            df_id=row['df_id'],
            row_number=row['row_number'],
            col_number=row['col_number'],
            row_index=row['row_index'],
            col_name=row['col_name'],
            index_names=row['index_names'],
            column_names=row['column_names'],
            value=row['value']
        )
    
    rows = list(map(serialize_row, flat_data))
    return spark.createDataFrame(rows)

def reconstruct_pandas_dataframes_from_spark(spark_df):
    pd_df = spark_df.toPandas()
    reconstructed = []

    for df_id in sorted(pd_df['df_id'].unique()):
        sub_df = pd_df[pd_df['df_id'] == df_id]
        index_names = sub_df['index_names'].iloc[0]
        column_names = sub_df['column_names'].iloc[0]

        sub_df['row_index'] = sub_df['row_index'].apply(tuple)
        sub_df['col_name'] = sub_df['col_name'].apply(tuple)

        pivot_df = sub_df.pivot(index='row_index', columns='col_name', values='value')
        
        if all(len(col) == 1 for col in pivot_df.columns):
            pivot_df.columns = [col[0] for col in pivot_df.columns]
        else:
            pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)
            pivot_df.columns.names = column_names

        pivot_df.index = pd.MultiIndex.from_tuples(pivot_df.index)
        pivot_df.index.names = index_names

        for col in pivot_df.columns:
            pivot_df[col] = pd.to_numeric(pivot_df[col], errors='ignore', downcast='integer')

        reconstructed.append(pivot_df)

    return reconstructed

# Example usage
spark = SparkSession.builder.appName("FlattenReconstruct").getOrCreate()

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
df2 = pd.DataFrame({'X': [5.1, 6.2], 'Y': [7.3, 8.4]}, index=[10, 11])
df_list = [df1, df2]

flat = flatten_pandas_dataframes(df_list)
spark_df = create_spark_dataframe_from_flat(spark, flat)

# Perform transformation
processed = spark_df.withColumn("value", col("value") * 10)

# Reconstruct DataFrames
reconstructed = reconstruct_pandas_dataframes_from_spark(processed)

# Show results
for i, df in enumerate(reconstructed):
    print(f"\nReconstructed DataFrame {i}:")
    print(df)
