#%%
import pandas as pd
import numpy as np

df = pd.read_csv('output/weebil_vespula.csv')

# %%

df['Spe'] = df['Spe'].replace(0,-1)
# %%
display(df)

df.to_csv('output/weebil_vespula_1-1.csv',sep=',',index=False)

# %%
