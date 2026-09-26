import pandas as pd
df = pd.read_csv('D:/MachineLearning/Random_shit/playground-series-s6e9/submission.csv')
print(f'Rows: {len(df)}')
print(f'Min: {df["Will_Buy_EV"].min():.6f}')
print(f'Max: {df["Will_Buy_EV"].max():.6f}')
print(f'Mean: {df["Will_Buy_EV"].mean():.6f}')
print(f'Unique values: {df["Will_Buy_EV"].nunique()}')