# -*- coding: utf-8 -*-
""" Originated in Google Colaboratory.
# Hierarchical Clustering single 、 average 、 complete
"""
import codecs
import itertools

import networkx as nx
import time
import matplotlib.pyplot as plt
import math
import os
import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmis
from scipy.cluster import hierarchy
from igraph import *
import igraph as ig
# Creates a NetworkX graph object
def make_graph(sim, labels=None):
    G = nx.Graph()
    edges = []
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
                    f.write(str(i) + delim + str(j) + "\r\n")
                else:
                    f.write("\"" + labels[i] + "\"" + delim + "\"" + labels[j] + "\"\r\n")
    print(filename + "  has Done")
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

# def export_communities(communities, label=None, filename='communities.csv', delim=",", header=True):
#     f = codecs.open(filename, 'w', 'utf-8')
#     if header:
#         f.write("Id,Community\n")
#     for i in range(len(communities)):
#         f.write("\"" + label[i] + "\"" + delim + str(communities[i]) + "\r\n")
#     print(filename + "  has Done")
#     f.close()


'''    # get place name list, semantic/content and score dictionary list '''
def get_PlaceName_SemanticScore(place365_result_csv):
    idx_places = []  # place's name list
    list_of_places__key_score = []  # places' semantic/content and score list
    # get original data in csv
    for root, sub_dir, files in os.walk(place365_result_csv):
        # only want to read files
        if sub_dir != []:  # Go to the bottom of the directory without folder
            continue
        # get places' name from sub_dir
        place_name = root.split('\\')[-1]
        idx_places.append(place_name)
        # get all key and score s of the place
        key_score_s = {}
        for file in files:
            with open(os.path.join(root, file), 'r') as f_tmp:
                score_key_list = f_tmp.read().split(',')
                for one_score_key in score_key_list[0:-1]:  # the last one is \n, not need it
                    score, key = one_score_key.split(':')
                    score = float(score)
                    if key in key_score_s:
                        key_score_s[key] += float(score)
                    else:
                        key_score_s[key] = float(score)
        list_of_places__key_score.append(key_score_s)
    # score normalization
    for i in range(len(list_of_places__key_score)):
        total_score = 0
        for _, score in list_of_places__key_score[i].items():
            total_score += score * score
        total_score = math.sqrt(total_score)
        for j in list_of_places__key_score[i]:
            list_of_places__key_score[i][j] /= total_score
    return idx_places, list_of_places__key_score

''' # calculate similarity matrix '''
def get_sim_mat(idx_places, list_of_places__key_score):
    num_place = len(idx_places)
    sim_mat = np.zeros((num_place, num_place))
    for i in range(num_place):
        place_A = list_of_places__key_score[i]
        for j in range(num_place):
            place_B = list_of_places__key_score[j]
            for k in place_A.keys():
                if k in place_B:
                    sim_mat[i][j] += place_A[k] * place_B[k]
    return sim_mat

''' # get distance matrix'''
def get_dis_mat(num_place, sim_mat):
    dis_mat = np.zeros((num_place, num_place))
    for i in range(num_place):
        for j in range(num_place):
            dis_mat[i][j] = 1 - sim_mat[i][j]
    return dis_mat

""" # Visualization include pals and Gephi """
def get_Visual_pals_Gephi(arch, linkage, cut_num, adjmat, cluster, idx_places):
    Visualization = ["pals", "Gephi"]
    for vis in Visualization:
        flag = False
        extension = ".dat"
        delim = ''
        if vis == "Gephi":
            flag = True
            extension = ".csv"
            delim = ','
        path = "HC_Visualization\\" + arch + "_HC_" + "results\\" + linkage + "\\" + vis + "\\"
        print(path)
        path = os.path.dirname(path)
        if not os.path.exists(path):  # check if exist path
            os.makedirs(path)  # if not create the folder
        dat_file = path + "\\" + arch + "-" + linkage + "-" + cut_num + "-" + vis + "-community" + extension
        # print("dat_file : " , dat_file)
        export_communities(cluster, idx_places, dat_file, delim=delim, header=flag)
        dat_file = path + "\\" + arch + "-" + linkage + "-" + cut_num + "-" + vis + "-edges" + extension
        export_edges(adjmat, idx_places, dat_file, delim=delim, header=flag)

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


""" # show dendrogram and export pals and Gephi """
# there are three models in  place365 but we only use "resnet50" by comparing
# architecture = ['resnet18', 'resnet50', 'alexnet']
architecture = ['resnet50']
for i, arch in enumerate(architecture):
    # init
    idx_places, list_of_places__key_score = get_PlaceName_SemanticScore(arch + '_result_csv')
    sim_mat = get_sim_mat(idx_places, list_of_places__key_score)
    # for i in sim_mat:
    #     for j in i:
    #         print(j, end=" ")
    #     print()
    num_place = len(idx_places)
    dis_mat = get_dis_mat(num_place, sim_mat)
    #
    # # show histogram to statistics
    # dissimflat = dis_mat.reshape((-1,))
    # _ = plt.hist(dissimflat[dissimflat != 0], bins=50)
    # plt.show()


    # remove some edge by threshold
    threshold = 0.52
    adjmat = dis_mat.reshape((-1,)).copy()
    adjmat[adjmat > threshold] = 0
    adjmat[adjmat > 0] = 1
    print(np.shape(adjmat))
    # for i in range(len(adjmat)):
    #     for j in range(len(adjmat[i])):
    #         print(adjmat[i][j], end=" ")
    #     print()
    print("{} out of {} edge values set to zero".format(len(adjmat[adjmat != 1]), len(adjmat)))
    adjmat = adjmat.reshape(dis_mat.shape)

    G = make_graph(adjmat, labels=idx_places)
    nx.draw_spring(G, with_labels=True)
    print("{} out of {} nodes left".format(len(G), len(adjmat)))
    print(G)
    from networkx.algorithms.community.centrality import girvan_newman

    comp = girvan_newman(G)
    # g1 = Graph(new_list)
    # print(comp)
    max_shown = 10
    shown_count = 1
    possibilities = []
    for communities in itertools.islice(comp, max_shown):
        # print("Possibility", shown_count, ": ", end='')
        # print(communities)
        # print("len(communities) : ", len(communities))
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
    # for which_possibility in range(max_shown):
    #     # which_possibility = 6
    #     communities = possibilities[which_possibility]
    #     num_communities = str(len(communities))
    #     print("num_communities : ", num_communities)

    which_possibility = 5

    communities = possibilities[which_possibility - 1]
    # print("len(communities) : ", len(communities))
    # print(communities)
    cut_num = len(communities)
    get_Visual_pals_Gephi(arch, "", str(cut_num), adjmat, communities, idx_places) # do pals and Gephi


    # nx.draw_spring(G, with_labels=True)
    # print("{} out of {} nodes left".format(len(G), len(adjmat)))
    # # plt.savefig('fig.png', bbox_inches='tight')
    # plt.show()

#     # Hierarchical Clustering including
#     linkage_enum = ['single', 'average', 'complete']
#     fig = plt.figure(figsize=(30, 18)) # for 3 dendrogram
#     plt.title("place365 model is " + arch + "\n\n", size=24)
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     for j, linkage in enumerate(linkage_enum):
#         ax = fig.add_subplot(1, 3, j + 1)
#         tmp = dis_mat
#         if linkage == 'complete':
#             tmp = sim_mat
#         t0 = time.time() # init time
#         Z = sch.linkage(tmp, method=linkage) # calculate and Z() is the result tree
#
#         # print("type", type(Z),"\n",Z)
#
#         # two way to get cluster results to pals and Gephi
#         # cluster = sch.fcluster(Z, t=1.01, criterion='inconsistent')
#         cut_num = 7
#         cluster = sch.cut_tree(Z, cut_num)
#         cluster = [i + 1 for x in cluster for i in x]
#         # print(cluster)
#         # print(type(cluster))
#         dn = hierarchy.dendrogram(Z, labels=idx_places, orientation='right', leaf_font_size=15) # do dendrogram
#         elapsed_time = time.time() - t0
#         ax.set_title('%s (time %.2fs)' % (linkage, elapsed_time), fontsize=18)
#
#         get_Visual_pals_Gephi(arch, linkage, str(cut_num), adjmat, cluster, idx_places) # do pals and Gephi
#         print()
#         # print results
#         print(arch + " Original cluster by " + linkage + " hierarchy clustering:")#, cluster, "\n"
#         print("     community number :", max(cluster))
#         # print(type(idx_places))
#         for d in idx_places:
#             print(d, end='\t')
#         print("\n")
#         for c in cluster:
#             print(c, end='\t')
#         print("\n")
#         dict = {}
#         for o, c in enumerate(cluster):
#             dict[c] = []
#         for o, c in enumerate(cluster):
#             dict[c].append(idx_places[o])
#         for key, value in dict.items():
#             print(len(dict[key]) , ':' ,dict[key])
#
#         '''
#         # compare similarity by mutual information
#         root = arch + "_NG\\array"
#         file = "new-7-labels_gn_pred.csv"
#         with open(os.path.join(root, file), 'r') as f_tmp:
#             tmp_list = f_tmp.read().split(',')
#
#         for c in tmp_list[0:-1]:
#             print(c, end='\t')
#         print("\n")
#         print(nmis(cluster, tmp_list[0:-1]))
#         # print("idx2place : ", idx_places)
#         print("\n")
#         '''
#
#     # plt.savefig(arch + ".png") # save picture if want
# plt.show()

