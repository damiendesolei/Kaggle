# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:38:25 2024

@author: damie
"""
import os
import glob
import pandas as pd

# Specify the folder containing the CSV files
folder_path = 'C:\\Users\\damie\\Dropbox\\Github\\Kaggle\\Classification_with_an_Academic_Success_Dataset\\others\\'

# Find all CSV files in the folder
all_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Load all CSV files into a list of DataFrames
dfs = [pd.read_csv(file) for file in all_files]

# Initialize an empty DataFrame for merging
merged_df = pd.DataFrame()

# Merge selected columns from each DataFrame
for i, df in enumerate(dfs):
    filename = os.path.basename(all_files[i])[:-4]  # Extract filename without extension
    merged_df[filename.replace(' ', '_')] = df['Target']

# Display the merged DataFrame
print(merged_df)


#submission
mode_per_row = merged_df.mode(axis=1)[0]





path = r'C:\Users\damie\Downloads\playground-series-s4e6\\'
test = pd.read_csv(path + 'test.csv', low_memory=True)

submission = pd.DataFrame()
submission['id'] = test['id']
submission['Target'] = mode_per_row.to_frame(name='Target')

submission.to_csv(path + 'submission.csv', index=False)
