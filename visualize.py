# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
# %%


def save_graph(fig, save_path):
    fig.savefig('./{}.png'.format(save_path), bbox_inches='tight')
    print('saved')


def draw_graph(df, save_path=None, is_save=False):
    linestyle = ["-", "--", ":", "-.", (0, (5, 10))]
    fig = plt.figure(figsize=(6, 6))
    for i in range(len(df.columns)):
        plt.plot(df.index,  df.iloc[:, i],
                 label=df.columns[i], linestyle=linestyle[i])

        plt.xlabel('Iteration Number', fontsize=18)
        plt.ylabel('AUC', fontsize=18)
        plt.legend()
    if is_save and save_graph != None:
        save_graph(fig, save_path)


def concat_df(*list):
    df = pd.concat(list, axis=1)
    df.columns = ['Omniscient', 'Random', 'Ours']
    return df


def sum_data_rat(data_list, spe=None):
    np_ans = np.zeros_like(np.loadtxt(spe + '/' + data_list[0], delimiter=','))
    for i in data_list:
        tmp = np.loadtxt(spe + '/' + i, delimiter=',')
        np_ans += tmp

    np_ans /= len(data_list)
    df = pd.DataFrame(np_ans)
    return df


def sum_data_wot(data_list, spe=None):
    np_ans = np.zeros_like(np.loadtxt(spe + '/' + data_list[0], delimiter=','))
    for i in data_list:
        tmp = np.loadtxt(spe + '/' + i, delimiter=',')
        np_ans += tmp

    np_ans /= len(data_list)
    df = pd.DataFrame(np_ans)
    return df


def sum_data_omt(data_list, spe=None):
    np_ans = np.zeros(500)

    for i in data_list:
        tmp = np.loadtxt(spe + '/' + i, delimiter=',')
        np_ans += tmp

    np_ans /= len(data_list)
    df = pd.Series(np_ans)
    return df


def sum_data_wj(data_list, spe=None):
    np_ans = np.zeros_like(np.loadtxt(spe+'/'+data_list[0], delimiter=','))
    for i in data_list:
        tmp = np.loadtxt(spe+'/'+i, delimiter=',')
        np_ans += tmp

    np_ans /= len(data_list)
    df = pd.DataFrame(np_ans)

    return df


def mean_data_wj(data_list, spe=None):
    new_df = pd.DataFrame()
    df = sum_data_wj(data_list, spe=spe)
    df = df.T
    df = df.sort_values(by=[0], ascending=True)
    new_df['0.75~'] = df[df.iloc[:, 0] > 0.75].mean(axis=0)
    new_df['0.65~0.75'] = df[df.iloc[:, 0] <=
                             0.75 and 0.65 < df.iloc[:, 0]].mean(axis=0)
    new_df['~0.65'] = df[df.iloc[:, 0] <= 0.65].mean(axis=0)
    # new_df['high'] = df.iloc[7:].mean(axis=0)
    # new_df['middle'] = df.iloc[3:7].mean(axis=0)
    # new_df['low'] = df.iloc[:3].mean(axis=0)
    return new_df


# %%
files = os.listdir()
files
# %%
wine, insect = 'wine', 'insect'
print(files)
wine_files = os.listdir(wine)
insect_files = os.listdir(insect)
# %%

insect_wot = []
insect_omt = []
insect_rat = []
for i in insect_files:
    name = list(i.split('_'))
    if name[4] in 'random.csv':
        # print('random')
        insect_rat.append(i)
    elif name[4] in 'omniscient.csv':
        # print('omni')
        insect_omt.append(i)
    else:
        # print('wot')
        insect_wot.append(i)

insect_rat_1 = []
insect_rat_2 = []
insect_rat_5 = []
insect_rat_10 = []
insect_rat_1_wj = []
insect_rat_2_wj = []
insect_rat_5_wj = []
insect_rat_10_wj = []

for i in insect_rat:
    name = list(i.split('_'))
    # print(name)
    if name[3] == '1':
        if 'wj' in i:
            insect_rat_1_wj.append(i)
        else:
            insect_rat_1.append(i)
    elif name[3] == "2":
        if 'wj' in i:
            insect_rat_2_wj.append(i)
        else:
            insect_rat_2.append(i)
    elif name[3] == "5":
        if 'wj' in i:
            insect_rat_5_wj.append(i)
        else:
            insect_rat_5.append(i)
    else:
        if 'wj' in i:
            insect_rat_10_wj.append(i)
        else:
            insect_rat_10.append(i)

insect_omt_1 = []
insect_omt_2 = []
insect_omt_5 = []
insect_omt_10 = []
insect_omt_1_wj = []
insect_omt_2_wj = []
insect_omt_5_wj = []
insect_omt_10_wj = []

for i in insect_omt:
    name = list(i.split('_'))
    # print(name)
    if name[3] == '1':
        if 'wj' in i:
            insect_omt_1_wj.append(i)
        else:
            insect_omt_1.append(i)
    elif name[3] == "2":
        if 'wj' in i:
            insect_omt_2_wj.append(i)
        else:
            insect_omt_2.append(i)
    elif name[3] == "5":
        if 'wj' in i:
            insect_omt_5_wj.append(i)
        else:
            insect_omt_5.append(i)
    else:
        if 'wj' in i:
            insect_omt_10_wj.append(i)
        else:
            insect_omt_10.append(i)

insect_wot_1 = []
insect_wot_2 = []
insect_wot_5 = []
insect_wot_10 = []
insect_wot_1_wj = []
insect_wot_2_wj = []
insect_wot_5_wj = []
insect_wot_10_wj = []

for i in insect_wot:
    name = list(i.split('_'))
    # print(name)
    if name[3] == '1':
        if 'wj' in i:
            insect_wot_1_wj.append(i)
        else:
            insect_wot_1.append(i)
    elif name[3] == "2":
        if 'wj' in i:
            insect_wot_2_wj.append(i)
        else:
            insect_wot_2.append(i)
    elif name[3] == "5":
        if 'wj' in i:
            insect_wot_5_wj.append(i)
        else:
            insect_wot_5.append(i)
    else:
        if 'wj' in i:
            insect_wot_10_wj.append(i)
        else:
            insect_wot_10.append(i)

# %%
df_insect_rat_1 = sum_data_rat(insect_rat_1, insect)
df_insect_omt_1 = sum_data_wot(insect_omt_1, insect)
df_insect_wot_1 = sum_data_wot(insect_wot_1, insect)
df_insect_1 = concat_df(df_insect_omt_1, df_insect_rat_1,
                        df_insect_wot_1.iloc[:, 2])

df_insect_rat_2 = sum_data_rat(insect_rat_2, insect)
df_insect_omt_2 = sum_data_omt(insect_omt_2, insect)
df_insect_wot_2 = sum_data_wot(insect_wot_2, insect)
df_insect_2 = concat_df(df_insect_omt_2, df_insect_rat_2,
                        df_insect_wot_2.iloc[:, 2])

df_insect_rat_5 = sum_data_rat(insect_rat_5, insect)
df_insect_omt_5 = sum_data_omt(insect_omt_5, insect)
df_insect_wot_5 = sum_data_wot(insect_wot_5, insect)
df_insect_5 = concat_df(df_insect_omt_5, df_insect_rat_5,
                        df_insect_wot_5.iloc[:, 2])

df_insect_rat_10 = sum_data_rat(insect_rat_10, insect)
df_insect_omt_10 = sum_data_omt(insect_omt_10, insect)
df_insect_wot_10 = sum_data_wot(insect_wot_10, insect)
df_insect_10 = concat_df(df_insect_omt_10, df_insect_rat_10,
                         df_insect_wot_10.iloc[:, 2])

# %%
# ここからwineだよ
# %%

wine_wot = []
wine_omt = []
wine_rat = []
for i in wine_files:
    name = list(i.split('_'))
    if name[4] in 'random.csv':
        # print('random')
        wine_rat.append(i)
    elif name[4] in 'omniscient.csv':
        # print('omni')
        wine_omt.append(i)
    else:
        # print('wot')
        wine_wot.append(i)

wine_rat_1 = []
wine_rat_2 = []
wine_rat_5 = []
wine_rat_10 = []
wine_rat_1_wj = []
wine_rat_2_wj = []
wine_rat_5_wj = []
wine_rat_10_wj = []

for i in wine_rat:
    name = list(i.split('_'))
    # print(name)
    if name[3] == '1':
        if 'wj' in i:
            wine_rat_1_wj.append(i)
        else:
            wine_rat_1.append(i)
    elif name[3] == "2":
        if 'wj' in i:
            wine_rat_2_wj.append(i)
        else:
            wine_rat_2.append(i)
    elif name[3] == "5":
        if 'wj' in i:
            wine_rat_5_wj.append(i)
        else:
            wine_rat_5.append(i)
    else:
        if 'wj' in i:
            wine_rat_10_wj.append(i)
        else:
            wine_rat_10.append(i)

wine_omt_1 = []
wine_omt_2 = []
wine_omt_5 = []
wine_omt_10 = []
wine_omt_1_wj = []
wine_omt_2_wj = []
wine_omt_5_wj = []
wine_omt_10_wj = []

for i in wine_omt:
    name = list(i.split('_'))
    # print(name)
    if name[3] == '1':
        if 'wj' in i:
            wine_omt_1_wj.append(i)
        else:
            wine_omt_1.append(i)
    elif name[3] == "2":
        if 'wj' in i:
            wine_omt_2_wj.append(i)
        else:
            wine_omt_2.append(i)
    elif name[3] == "5":
        if 'wj' in i:
            wine_omt_5_wj.append(i)
        else:
            wine_omt_5.append(i)
    else:
        if 'wj' in i:
            wine_omt_10_wj.append(i)
        else:
            wine_omt_10.append(i)

wine_wot_1 = []
wine_wot_2 = []
wine_wot_5 = []
wine_wot_10 = []
wine_wot_1_wj = []
wine_wot_2_wj = []
wine_wot_5_wj = []
wine_wot_10_wj = []

for i in wine_wot:
    name = list(i.split('_'))
    # print(name)
    if name[3] == '1':
        if 'wj' in i:
            wine_wot_1_wj.append(i)
        else:
            wine_wot_1.append(i)
    elif name[3] == "2":
        if 'wj' in i:
            wine_wot_2_wj.append(i)
        else:
            wine_wot_2.append(i)
    elif name[3] == "5":
        if 'wj' in i:
            wine_wot_5_wj.append(i)
        else:
            wine_wot_5.append(i)
    else:
        if 'wj' in i:
            wine_wot_10_wj.append(i)
        else:
            wine_wot_10.append(i)

# %%
df_wine_rat_1 = sum_data_rat(wine_rat_1, wine)
df_wine_omt_1 = sum_data_wot(wine_omt_1, wine)
df_wine_wot_1 = sum_data_wot(wine_wot_1, wine)
df_wine_1 = concat_df(df_wine_omt_1, df_wine_rat_1,
                      df_wine_wot_1.iloc[:, 2])

df_wine_rat_2 = sum_data_rat(wine_rat_2, wine)
df_wine_omt_2 = sum_data_omt(wine_omt_2, wine)
df_wine_wot_2 = sum_data_wot(wine_wot_2, wine)
df_wine_2 = concat_df(df_wine_omt_2, df_wine_rat_2,
                      df_wine_wot_2.iloc[:, 2])

df_wine_rat_5 = sum_data_rat(wine_rat_5, wine)
df_wine_omt_5 = sum_data_omt(wine_omt_5, wine)
df_wine_wot_5 = sum_data_wot(wine_wot_5, wine)
df_wine_5 = concat_df(df_wine_omt_5, df_wine_rat_5,
                      df_wine_wot_5.iloc[:, 2])

df_wine_rat_10 = sum_data_rat(wine_rat_10, wine)
df_wine_omt_10 = sum_data_omt(wine_omt_10, wine)
df_wine_wot_10 = sum_data_wot(wine_wot_10, wine)
df_wine_10 = concat_df(df_wine_omt_10, df_wine_rat_10,
                       df_wine_wot_10.iloc[:, 2])

# %%
# 描画していくよ
draw_graph(df_insect_1, 'insect_1', is_save=True)
draw_graph(df_insect_2, 'insect_2', is_save=True)
draw_graph(df_insect_5, 'insect_5', is_save=True)
draw_graph(df_insect_10, 'insect_10', is_save=True)
# %%
draw_graph(df_wine_1, 'wine_1', is_save=True)
draw_graph(df_wine_2, 'wine_2', is_save=True)
draw_graph(df_wine_5, 'wine_5', is_save=True)
draw_graph(df_wine_10, 'wine_10', is_save=True)


# %%
# wj をしらべていくよ

# %%

df_insect_wot_1_wj = mean_data_wj(insect_wot_1_wj, spe='insect')
df_insect_wot_2_wj = mean_data_wj(insect_wot_2_wj, spe='insect')
df_insect_wot_5_wj = mean_data_wj(insect_wot_5_wj, spe='insect')
df_insect_wot_10_wj = mean_data_wj(insect_wot_10_wj, spe='insect')

df_wine_wot_1_wj = mean_data_wj(wine_wot_1_wj, spe='wine')
df_wine_wot_2_wj = mean_data_wj(wine_wot_2_wj, spe='wine')
df_wine_wot_5_wj = mean_data_wj(wine_wot_5_wj, spe='wine')
df_wine_wot_10_wj = mean_data_wj(wine_wot_10_wj, spe='wine')

draw_graph(df_insect_wot_1_wj, 'insect_wot_1_wj', is_save=True)
draw_graph(df_insect_wot_2_wj, 'insect_wot_2_wj', is_save=True)
draw_graph(df_insect_wot_5_wj, 'insect_wot_5_wj', is_save=True)
draw_graph(df_insect_wot_10_wj, 'insect_wot_10_wj', is_save=True)


draw_graph(df_wine_wot_1_wj, 'wine_wot_1_wj', is_save=True)
draw_graph(df_wine_wot_2_wj, 'wine_wot_2_wj', is_save=True)
draw_graph(df_wine_wot_5_wj, 'wine_wot_5_wj', is_save=True)
draw_graph(df_wine_wot_10_wj, 'wine_wot_10_wj', is_save=True)


# %%
df_wine_wot_1[2].idxmax()

# %%

df_insect_omt_1_wj = mean_data_wj(insect_omt_1_wj, spe='insect')
df_insect_omt_2_wj = mean_data_wj(insect_omt_2_wj, spe='insect')
df_insect_omt_5_wj = mean_data_wj(insect_omt_5_wj, spe='insect')
df_insect_omt_10_wj = mean_data_wj(insect_omt_10_wj, spe='insect')

df_wine_omt_1_wj = mean_data_wj(wine_omt_1_wj, spe='wine')
df_wine_omt_2_wj = mean_data_wj(wine_omt_2_wj, spe='wine')
df_wine_omt_5_wj = mean_data_wj(wine_omt_5_wj, spe='wine')
df_wine_omt_10_wj = mean_data_wj(wine_omt_10_wj, spe='wine')

draw_graph(df_insect_omt_1_wj, 'insect_omt_1_wj', is_save=True)
draw_graph(df_insect_omt_2_wj, 'insect_omt_2_wj', is_save=True)
draw_graph(df_insect_omt_5_wj, 'insect_omt_5_wj', is_save=True)
draw_graph(df_insect_omt_10_wj, 'insect_omt_10_wj', is_save=True)


draw_graph(df_wine_omt_1_wj, 'wine_omt_1_wj', is_save=True)
draw_graph(df_wine_omt_2_wj, 'wine_omt_2_wj', is_save=True)
draw_graph(df_wine_omt_5_wj, 'wine_omt_5_wj', is_save=True)
draw_graph(df_wine_omt_10_wj, 'wine_omt_10_wj', is_save=True)

# %%

df_insect_rat_1_wj = mean_data_wj(insect_rat_1_wj, spe='insect')
df_insect_rat_2_wj = mean_data_wj(insect_rat_2_wj, spe='insect')
df_insect_rat_5_wj = mean_data_wj(insect_rat_5_wj, spe='insect')
df_insect_rat_10_wj = mean_data_wj(insect_rat_10_wj, spe='insect')

df_wine_rat_1_wj = mean_data_wj(wine_rat_1_wj, spe='wine')
df_wine_rat_2_wj = mean_data_wj(wine_rat_2_wj, spe='wine')
df_wine_rat_5_wj = mean_data_wj(wine_rat_5_wj, spe='wine')
df_wine_rat_10_wj = mean_data_wj(wine_rat_10_wj, spe='wine')

draw_graph(df_insect_rat_1_wj, 'insect_rat_1_wj', is_save=True)
draw_graph(df_insect_rat_2_wj, 'insect_rat_2_wj', is_save=True)
draw_graph(df_insect_rat_5_wj, 'insect_rat_5_wj', is_save=True)
draw_graph(df_insect_rat_10_wj, 'insect_rat_10_wj', is_save=True)


draw_graph(df_wine_rat_1_wj, 'wine_rat_1_wj', is_save=True)
draw_graph(df_wine_rat_2_wj, 'wine_rat_2_wj', is_save=True)
draw_graph(df_wine_rat_5_wj, 'wine_rat_5_wj', is_save=True)
draw_graph(df_wine_rat_10_wj, 'wine_rat_10_wj', is_save=True)

# %%
# 箱ひげ図を書いていくよ


def concat_df_for_boxplot(df, spe=None):
    tmp = []
    for i in df:
        a = pd.read_csv(spe + '/' + i, header=None)
        tmp.extend(a.iloc[0, :])
    return tmp


def return_df_for_boxplot(l1=None, l2=None, l5=None, l10=None, spe=None):
    if spe is None:
        raise ValueError
    l1_list = concat_df_for_boxplot(l1, spe)
    l2_list = concat_df_for_boxplot(l2, spe)
    l5_list = concat_df_for_boxplot(l5, spe)
    l10_list = concat_df_for_boxplot(l10, spe)
    df = pd.DataFrame(
        {'1': l1_list,
         '2': l2_list,
         '5': l5_list,
         '10': l10_list}
    )
    return df


def draw_boxplot(df, save_path=None, is_save=False):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Initial AUC', fontsize=18)
    ax.set_ylabel('lambda', fontsize=18)
    sns.boxplot(data=df,
                orient='h',
                palette='Pastel1',
                width=0.5,
                sym=''
                )
    if save_path is not None and is_save == True:
        save_graph(fig, save_path)


# %%
insect_box = return_df_for_boxplot(insect_omt_1_wj, insect_omt_2_wj,
                                   insect_omt_5_wj, insect_omt_10_wj, spe='insect')
wine_box = return_df_for_boxplot(wine_omt_1_wj, wine_omt_2_wj,
                                 wine_omt_5_wj, wine_omt_10_wj, spe='wine')
draw_boxplot(insect_box, save_path='insect_box', is_save=True)
draw_boxplot(wine_box, save_path='wine_box', is_save=True)

# %%
