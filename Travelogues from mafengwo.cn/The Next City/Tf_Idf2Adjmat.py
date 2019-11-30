'''
TF_IDF 用于获得城市的邻接矩阵
    1. 对分词清洗后的城市游记计算每个词的TF-IDF值
    2. 以余弦相似度作为城市游记文本相似度度量标准，将 1 减相似度作为距离计算邻接矩阵，结果保存在 AdjMat.csv 中
'''
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jieba
# import gensim
# from gensim import corpora,similarities,models
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image
import numpy as np
import itertools
import math
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize


base = 'yj_city_content_wash_20777'
files = os.listdir(base)
name = re.compile("(.*)_wash.txt")
cities = [] # 获取城市名 即 label
texts = []
for file in files:
    with open(os.path.join(base,file),'r',encoding="utf-8")as f:
        data = f.read().split(' ')
        data = [word for word in data if (word!=' ' and word !='\xa0' and word != '' and word!='★' and word!='●')] # \xa0 是不间断空白符 &nbsp;
    texts.append(data)
    city = name.findall(file)
    cities.append(*city)

# 语料
corpus = [' '.join(text) for text in texts]
# f = open("corpus_me.txt", 'w',  encoding="utf-8")
# for c in corpus:
#     f.write(c + "\r\n")
# f.close()


# TD-IDF
vectorizer=CountVectorizer(min_df=5,max_df=1.0, ngram_range=(1,3))
tfidf_transformer = TfidfTransformer()
# 将文本中的词语转换为词频矩阵
vectorizer.fit(corpus)
X = vectorizer.transform(corpus)
# 将词频矩阵X统计成TF-IDF值矩阵
tfidf_transformer.fit(X.toarray())
tfidf = tfidf_transformer.transform(X)
# np.savetxt('_me.txt',tfidf.toarray())

#
# 将稀疏矩阵转化为稠密矩阵 全部输出
city_tfidf_mat = pd.DataFrame(tfidf.todense(),index=cities, columns=vectorizer.get_feature_names())
# city_tfidf_mat.to_csv('me.txt')
# 归一化
mms = MinMaxScaler()
# 将属性缩放到（0-1）之间
norm_city_tfidf_mat = mms.fit_transform(city_tfidf_mat)
# # 保存一下，看下结果
feature= pd.DataFrame(norm_city_tfidf_mat,index=cities, columns=vectorizer.get_feature_names())
# 去掉只有一个值得属性
de = []
for col in feature.columns:
    cnt = 0
    for row in range(0, len(feature)):
        if feature.iloc[row][col] - 0 == 0:
            cnt += 1
    # print(col, " : ", cnt)
    if cnt < 2:
        de.append(col)
for item in de:
    feature.drop(item,axis=1)
# feature.to_excel('feature.xlsx')
# f = open("result.txt", "w", encoding="utf-8")
# for item in vectorizer.get_feature_names():
#     f.write(item + "\n")
# f.close()
# # print('!!!\n', vectorizer.get_feature_names())
#
#
# 余弦相似度 求 相似矩阵
sim = cosine_similarity(feature, feature)
# CSV_sim = pd.DataFrame(sim,columns=cities, index=cities)
# CSV_sim.to_csv("sim.csv")

# # Analyze distribution of dissimilarity score
# simflat = sim.reshape((-1,))
# simflat = simflat[simflat != 1.] # Too many ones result in a bad histogram so we remove them
# print(simflat)
# _ = plt.hist(simflat, bins=25)
# plt.show()
# mmax  = np.max(simflat)
# mmin  = np.min(simflat)
# mmean = np.mean(simflat)
# print('avg={0:.2f} min={1:.2f} max={2:.2f}'.format(mmean, mmin, mmax))

# 经实验 threshold 取 0.16 才能保证每个城市至少与另一座城市有边相连
# 经实测 threshold 取 0.20 社团效果较好

# # 构建邻接矩阵
threshold = 0.20 # Do not change this value
adjmat = sim.copy()
np.fill_diagonal(adjmat, np.min(sim)) # 去掉自环
adjmat = adjmat.reshape((-1,))
adjmat[adjmat < threshold] = 1
adjmat = 1. - adjmat
print("{} out of {} values set to zero".format(len(adjmat[adjmat == 0]), len(adjmat)))
adjmat = adjmat.reshape(sim.shape)

# flag = [0 for i in range(adjmat.shape[0])]
for i in range(adjmat.shape[0]):
    flag = (adjmat[i] != 0) * 1.0
    # print(np.sum(flag == 0))
    if np.sum(flag == 0) >= 50:
        print(cities[i])

CSV_adj = pd.DataFrame(adjmat,columns=cities, index=cities)
# print(CSV_adj)
CSV_adj.to_csv("AdjMat.csv")
