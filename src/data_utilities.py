import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def df_analysis(df, feature):
    # shows a data analysis with an histogram and a vertical graph of the dataframe
    # :param df: dataframe
    # :param feature: the feature to analyze

    f, axes = plt.subplots(1, 2, figsize=(10, 8))
    sns.histplot(data=df[feature], ax=axes[0])
    sns.boxplot(data=df[feature], ax=axes[1])
    plt.show()


def col_analisys(df, col):
    # shows the number of values of a feature
    # :param df: dataframe
    # :param col: feature name
    print(df[col].value_counts())


def corr_matrix(df):

    # shows the correlation matrix of a dataframe
    # :param df:

    plt.figure(figsize=(15, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.show()


def pair_plot(df):
    # shows the pair plot of a dataframe
    # :param df: dataframe

    sns.pairplot(df)
    plt.show()


def box_plot(df):
    # shows the box plot of a dataframe
    # :param df: dataframe

    plt.figure(figsize=(15, 4))
    sns.boxplot(data=df, orient="h")
    plt.show()
