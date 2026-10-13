import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def corr(df: pd.DataFrame):
    corr = df.corr()

    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False,
        annot = True
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        horizontalalignment="right"
    )

    plt.tight_layout()

def scatter(x, y, cols):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, col in zip(axes.flat, cols):
        sns.regplot(x=x[col], y=y, ax=ax, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        ax.set_title(col)

    plt.tight_layout()

def histogram(x, cols):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for ax, col in zip(axes.flat, cols):
        sns.histplot(x=x[col], ax=ax)
        ax.set_title(col)

    plt.tight_layout()

def show():
    plt.show()