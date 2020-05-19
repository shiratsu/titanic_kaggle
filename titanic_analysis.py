# -*- coding: utf-8 -*-

import numpy 
import pandas as pd

import os
path=os.getcwd()
print(path)

##############################
# データ前処理 1
# 必要な項目を抽出する
##############################

train = pd.read_csv("csv/train.csv")
print(train.head())

test = pd.read_csv('csv/test.csv')
print(test.head())