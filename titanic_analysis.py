# -*- coding: utf-8 -*-

import numpy
import pandas as pd

import os
import matplotlib.pyplot as plt

path = os.getcwd()
print(path)

##############################
# データ前処理 1
# 必要な項目を抽出する
##############################

train = pd.read_csv("csv/train.csv")
print(train.head())

# 一応基本統計量も確認しておく
print(train.describe())

# sibspとParchで集計してみる
# print("test")
# print(train.query('Survived == 1'))
# print("test")

# そもそも兄弟いる人と、子供の数を確認
# print(train.groupby(["SibSp"]).count())
# print(train.groupby(["Parch"]).count())


# 以下いろいろ計算してみたけど、多分兄弟や配偶者の乗船、子供の乗船がゼロの方が多いので、あまり意味ないかも
suv0 = train.query('Survived == 0')
suv1 = train.query('Survived == 1')

# 乗船してる兄弟の人数で生き残れてるのかどうか確認する
suv0BySibSp = suv0.groupby(["SibSp"]).count()[
    ["Survived"]]
# print(suv0BySibSp)

suv1BySibSp = suv1.groupby(["SibSp"]).count()[
    ["Survived"]]
# print(suv1BySibSp)

test = pd.read_csv('csv/test.csv')
# print(test.head())

# 必要な項目のみ抽出
df_train = train.loc[:, ['Survived', 'Pclass',
                         'Sex', 'Age', 'Fare', 'Embarked']]
print(df_train.head())

##############################
# データ前処理 2
# 欠損値を処理する
##############################
# 現在の長さ
print("##現在のデータの長さ##")
print(len(df_train))

# 欠損値がないか確認する
print("##欠損値の合計##")
print(df_train.isnull().sum())

# 年齢がnullの行を削除する(わからないものはわからないので)
# Delete rows with null age
df_train = df_train.dropna(subset=['Age']).reset_index(drop=True)
print("##null削除後の長さ##")
print(len(df_train))


# 欠損を探す
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


kesson_table(df_train)
kesson_table(test)

# 欠損を中央値で埋めると言う考えもある(今回はしない)
# df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())

# 平均値で埋めると言うやり方もある
#df_train["Age"].fillna(df_train.Age.mean(), inplace=True)
#df_train["Embarked"].fillna(df_train.Embarked.mean(), inplace=True)


print("##null削除後の長さ##")
print(len(df_train))

# 客室のグレードごとのHistgramを出してみる
split_data = []
for survived in [0, 1]:
    split_data.append(df_train[df_train.Survived == survived])

temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3)
plt.show()

# 次は年齢ごとにヒストグラムを出してみます
temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16)
plt.show()


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train.head(10)

# カテゴリデータを数値化
df_train["Sex"][df_train["Sex"] == "male"] = 0
df_train["Sex"][df_train["Sex"] == "female"] = 1
df_train["Embarked"][df_train["Embarked"] == "S"] = 0
df_train["Embarked"][df_train["Embarked"] == "C"] = 1
df_train["Embarked"][df_train["Embarked"] == "Q"] = 2
