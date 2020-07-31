# %%
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import random
sns.set()
np.random.seed(28)
random.seed(28)
# %%
ves_mu = np.array([0.10, 0.13])
ves_sigma = np.array([[0.12, 0], [0, 0.12]])

val_ves = np.random.multivariate_normal(ves_mu, ves_sigma, 100)
sns.jointplot(val_ves[:, 0], val_ves[:, 1])
# val_ves = val_ves[val_ves[:, 0] + val_ves[:, 1] > 0]
val_ves[:, 0] = np.exp([val_ves[:, 0]])

plt.show()


# %%
wee_mu = np.array([-0.10, -0.13])
wee_sigma = np.array([[0.12, 0], [0, 0.12]])

val_wee = np.random.multivariate_normal(wee_mu, wee_sigma, 100)
sns.jointplot(val_wee[:, 0], val_wee[:, 1])
# val_wee = val_wee[val_wee[:, 0] + val_wee[:, 1] < 0]
val_wee[:, 0] = np.exp(val_wee[:, 0])
plt.show()

# %%'
sns.set()
df_wee = pd.DataFrame(val_wee)
df_wee['Species'] = 'Weevil'
df_ves = pd.DataFrame(val_ves)
df_ves['Species'] = 'Vespula'

df_all = pd.concat([df_wee, df_ves], axis=0)
sns.scatterplot(data=df_all, x=0, y=1, hue='Species')
print('{}{}{}'.format('-'*10, 'weebil', '-'*10))
print(df_wee.describe())
print('{}{}{}'.format('-'*10, 'vespula', '-'*10))
print(df_ves.describe())
print('{}{}{}'.format('-' * 10, 'all', '-' * 10))
print(df_all.describe())

# %%
# vesの画像生成
for index, row in df_ves.iterrows():
    im = Image.new(mode='L', size=(512, 512), color=255)
    draw = ImageDraw.Draw(im)
    center_x = im.size[0] / 2 + random.randint(-20, 20)
    center_y = im.size[1] / 2 + random.randint(-20, 20)
    body_width, body_height = 200, 100
    head_width, head_hight = body_height * \
        row[0], body_width * row[0]

    body_lu_x = center_x - body_width
    body_lu_y = center_y - body_height / 2
    body_rd_x = center_x
    body_rd_y = center_y + body_height / 2

    head_lu_x = center_x
    head_lu_y = center_y - head_hight / 2
    head_rd_x = center_x + head_width
    head_rd_y = center_y + head_hight / 2

    diff_x = 40
    diff_y = 40
    foot_cu_x = (center_x - body_lu_x) / 2 + body_lu_x
    foot_cu_y = body_lu_y
    foot_cd_x = foot_cu_x
    foot_cd_y = body_rd_y

    foot_lu_x = foot_cu_x - diff_x
    foot_lu_y = foot_cu_y - diff_y
    foot_ld_x = foot_cd_x - diff_x
    foot_ld_y = foot_cd_y + diff_y

    draw.line((foot_lu_x, foot_lu_y, foot_cu_x, foot_cu_y), fill=0, width=10)
    draw.line((foot_ld_x, foot_ld_y, foot_cd_x, foot_cd_y), fill=0, width=10)
    draw.ellipse(
        (body_lu_x, body_lu_y, body_rd_x,
         body_rd_y), fill=128 + random.randint(-20, 20)
    )
    draw.ellipse(
        (body_lu_x, body_lu_y, body_rd_x,
         body_rd_y), fill=128 + random.randint(-20, 20)
    )

    draw.ellipse(
        (head_lu_x, head_lu_y, head_rd_x, head_rd_y),
        fill=int(128 + int(150*row[1]))
    )
    im.save(
        'img/{0}/{1}_{2}.jpg'.format(row['Species'], row['Species'], str(index).zfill(3)))


# %%
# weebilの画像生成
for index, row in df_wee.iterrows():
    im = Image.new(mode='L', size=(512, 512), color=255)
    draw = ImageDraw.Draw(im)
    center_x = im.size[0] / 2 + random.randint(-20, 20)
    center_y = im.size[1] / 2 + random.randint(-20, 20)
    body_width, body_height = 200, 100
    head_width, head_hight = body_height * \
        row[0], body_width * row[0]

    body_lu_x = center_x - body_width
    body_lu_y = center_y - body_height / 2
    body_rd_x = center_x
    body_rd_y = center_y + body_height / 2

    head_lu_x = center_x
    head_lu_y = center_y - head_hight / 2
    head_rd_x = center_x + head_width
    head_rd_y = center_y + head_hight / 2

    diff_x = 40
    diff_y = 40
    foot_cu_x = (center_x - body_lu_x) / 2 + body_lu_x
    foot_cu_y = body_lu_y
    foot_cd_x = foot_cu_x
    foot_cd_y = body_rd_y

    foot_lu_x = foot_cu_x - diff_x
    foot_lu_y = foot_cu_y - diff_y
    foot_ld_x = foot_cd_x - diff_x
    foot_ld_y = foot_cd_y + diff_y

    draw.line((foot_lu_x, foot_lu_y, foot_cu_x, foot_cu_y), fill=0, width=10)
    draw.line((foot_ld_x, foot_ld_y, foot_cd_x, foot_cd_y), fill=0, width=10)
    draw.ellipse(
        (body_lu_x, body_lu_y, body_rd_x,
         body_rd_y), fill=128 + random.randint(-20, 20)
    )
    draw.ellipse(
        (body_lu_x, body_lu_y, body_rd_x,
         body_rd_y), fill=128 + random.randint(-20, 20)
    )

    draw.ellipse(
        (head_lu_x, head_lu_y, head_rd_x, head_rd_y),
        fill=int(128 + int(150 * row[1]))
    )

    im.save(
        'img/{0}/{1}_{2}.jpg'.format(row['Species'], row['Species'], str(index).zfill(3)))


# %%
for index, row in df_all.iterrows():
    im = Image.new(mode='L', size=(512, 512), color=255)
    draw = ImageDraw.Draw(im)
    center_x = im.size[0] / 2 + random.randint(-20, 20)
    center_y = im.size[1] / 2 + random.randint(-20, 20)
    body_width, body_height = 200, 100
    head_width, head_hight = body_height * \
        row[0]/2, body_width * row[0]/2

    body_lu_x = center_x - body_width
    body_lu_y = center_y - body_height / 2
    body_rd_x = center_x
    body_rd_y = center_y + body_height / 2

    head_lu_x = center_x
    head_lu_y = center_y - head_hight / 2
    head_rd_x = center_x + head_width
    head_rd_y = center_y + head_hight / 2

    diff_x = 40
    diff_y = 40
    foot_cu_x = (center_x - body_lu_x) / 2 + body_lu_x
    foot_cu_y = body_lu_y
    foot_cd_x = foot_cu_x
    foot_cd_y = body_rd_y

    foot_lu_x = foot_cu_x - diff_x
    foot_lu_y = foot_cu_y - diff_y
    foot_ld_x = foot_cd_x - diff_x
    foot_ld_y = foot_cd_y + diff_y

    draw.line((foot_lu_x, foot_lu_y, foot_cu_x, foot_cu_y), fill=0, width=10)
    draw.line((foot_ld_x, foot_ld_y, foot_cd_x, foot_cd_y), fill=0, width=10)
    draw.ellipse(
        (body_lu_x, body_lu_y, body_rd_x,
         body_rd_y), fill=128 + random.randint(-20, 20)
    )

    draw.ellipse(
        (head_lu_x, head_lu_y, head_rd_x, head_rd_y),
        fill=int(128 + int(150 * row[1]))
    )

    im.save(
        'img/{0}/{1}_{2}.jpg'.format('all', row['Species'], str(index).zfill(3)))
# %%


def make_data():
    ves_mu = np.array([0.10, 0.13])
    ves_sigma = np.array([[0.12, 0], [0, 0.12]])

    val_ves = np.random.multivariate_normal(ves_mu, ves_sigma, 1000)
    sns.jointplot(val_ves[:, 0], val_ves[:, 1])
    # val_ves = val_ves[val_ves[:, 0] + val_ves[:, 1] > 0]
    # val_ves[:, 0] = np.exp([val_ves[:, 0]])

    plt.show()

    wee_mu = np.array([-0.10, -0.13])
    wee_sigma = np.array([[0.12, 0], [0, 0.12]])

    val_wee = np.random.multivariate_normal(wee_mu, wee_sigma, 1000)
    sns.jointplot(val_wee[:, 0], val_wee[:, 1])
    # val_wee = val_wee[val_wee[:, 0] + val_wee[:, 1] < 0]
    # val_wee[:, 0] = np.exp(val_wee[:, 0])
    plt.show()

    sns.set()
    df_wee = pd.DataFrame(val_wee)
    df_wee['Species'] = 1
    df_ves = pd.DataFrame(val_ves)
    df_ves['Species'] = 0

    # weebil 1 vespula 0

    df_all = pd.concat([df_wee, df_ves], axis=0)
    sns.scatterplot(data=df_all, x=0, y=1, hue='Species')
    print('{}{}{}'.format('-'*10, 'weebil', '-'*10))
    print(df_wee.describe())
    print('{}{}{}'.format('-'*10, 'vespula', '-'*10))
    print(df_ves.describe())
    print('{}{}{}'.format('-' * 10, 'all', '-' * 10))
    print(df_all.describe())

    return df_all
# %%


def main():
    df_all = make_data()
    df_all = df_all.rename(columns={0: 'f1', 1: 'f2'})
    df_all.to_csv('output/weebil_vespula.csv', index=False)


main()

# %%
if __name__ == "__main__":
    main()
# %%
display(df_all)
sns.set_style('white')
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax = sns.scatterplot(x=0, y=1, hue='Species', data=df_all)
# ax.set(xlabel='the  head/body  size  ratio', ylabel='the head/body color ratio')
ax.set_xlabel(xlabel='the  head/body size ratio',fontsize=18)
ax.set_ylabel(ylabel='the  head/body color ratio',fontsize=18)

ax.legend(fontsize=16)


x = np.linspace(-1, 3)
y = (-1.75371675 / 2.42340284) * x + 0.7

ax.plot(x, y)

plt.tick_params(labelsize=16)
fig.savefig('scatterplot.png',bbox_inches='tight')
fig.savefig('scatterplot.eps',bbox_inches='tight')
fig.savefig('scatterplot.pdf',bbox_inches='tight')
# -1.75371675 -2.42340284
# %%
