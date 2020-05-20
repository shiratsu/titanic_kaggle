# -*- coding: utf-8 -*-

import numpy
import pandas as pd

import os
path = os.getcwd()
print(path)

##############################
# データ前処理 1
# 必要な項目を抽出する
##############################

train = pd.read_csv("csv/train.csv")
print(train.head())

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
df_train = train.loc[:, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
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
