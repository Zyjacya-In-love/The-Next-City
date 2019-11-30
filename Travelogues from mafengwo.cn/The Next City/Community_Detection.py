'''
Community_Detection 用于社团发现
    1. Girvan-Newman算法
    2. 输出 Gephi 格式
'''
import codecs

import pandas as pd
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.metrics import jaccard_similarity_score
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import networkx as nx
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image
import numpy as np
import itertools
import math
import os

# Creates a NetworkX graph object
def make_graph(sim, labels=None):
    G = nx.Graph()
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            if i != j and sim[i, j] != 0:
                if labels == None:
                    G.add_edge(i, j, weight=sim[i, j])
                else:
                    G.add_edge(labels[i], labels[j], weight=sim[i, j])
    return G


# Save graph for use in Gephi or pals
def export_edges(sim, labels=None, filename="edges.csv", delim=",", header=True):
    f = codecs.open(filename, 'w', 'utf-8')
    if header:
        f.write("Source,Target\n")
    for i in range(sim.shape[0]):
        for j in range(i + 1, sim.shape[1]):
            if sim[i, j] != 0:
                if labels == None:
                    f.write(str(i) + delim + str(j) + "\n")
                else:
                    f.write("\"" + labels[i] + "\"" + delim + "\"" + labels[j] + "\"\n")
    f.close()


def export_communities(communities, label, filename='communities.csv', delim=",", header=True):
    indices_in_community = []

    f = codecs.open(filename, 'w', 'utf-8')
    if header:
        f.write("Id,Community\n")
    cur_com = 1
    for c in communities:
        indices = [i for i, x in enumerate(label) if x in c]
        indices_in_community.extend(indices)
        for i in indices:
            f.write("\"" + label[i] + "\"" + delim + str(cur_com) + "\r\n")
        cur_com += 1
    f.close()


class Config():
    colors = ['aquamarine', 'bisque', 'blanchedalmond', 'blueviolet', 'brown',
              'burlywood', 'cadetblue', 'chartreuse','chocolate', 'coral',
              'cornflowerblue', 'cornsilk', 'crimson', 'darkblue', 'darkcyan',
              'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
              'darkmagenta', 'darkolivegreen', 'darkorange', 'darkslateblue',
              'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
              'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
              'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue',
              'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
              'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
              'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory']
    labels = None

dataset = pd.read_csv('AdjMat.csv',sep=',') # , index_col=0,header=0 声明文件第一列为索引，第一行为列名
adjmat = dataset.iloc[:,1:].values
cities = dataset.iloc[:,0].values
cities = list(cities)
Config.labels = cities

# 依照邻接矩阵构建网络
G = make_graph(adjmat, labels=Config.labels)
# nx.draw(G, with_labels=True)
# plt.show()

# Girvan-Newman算法
comp = girvan_newman(G)

max_shown = 10
shown_count = 1
possibilities = []
for communities in itertools.islice(comp, max_shown):
    print("Possibility", shown_count, ": ", end='')
    print(communities)
    print("len(communities) : ", len(communities))
    print()
    possibilities.append(communities)
    color_map = ["" for x in range(len(G))]
    color = 0
    for c in communities:
        indices = [i for i, x in enumerate(G.nodes) if x in c]
        for i in indices:
            color_map[i] = Config.colors[color]
        color += 2
    shown_count += 1
    # nx.draw(G, node_color=color_map, with_labels=True)
    # plt.show()


# 生成可视化文件
which_possibility = 4
communities = possibilities[which_possibility-1]
extension = ".csv"
delim = ','
path = "Visualization\\"
if not os.path.exists(path):  # check if exist path
    os.makedirs(path)  # if not create the folder
cfile = path + "community" + extension
print(cfile)
export_communities(communities, cities, cfile, delim=delim)
efile = path + "edges" + extension
export_edges(adjmat, labels=cities, filename=efile, delim=delim)
