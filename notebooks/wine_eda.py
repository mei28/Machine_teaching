# %%
import pandas as pd

# %%
df = pd.read_csv('../output/winequality-red.csv', sep=';')

# %%
display(df)
df.describe()
# %%
df['quality'].describe()
#%%
df['quality'] = df['quality'].apply(lambda x: 1 if x > 5 else -1)


# %%
df['quality'].describe()

# %%
df.shape
df.describe()

# %%
df.to_csv('../output/wine-quality-pm1.csv')