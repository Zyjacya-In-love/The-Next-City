'''
WashContent 用于清洗城市游记
    1. 将每个城市的所有游记利用 结巴 进行分词，并去停止词，将清洗结果存到 yj_city_content 目录下
    2. 清洗后的游记内容依据词频得到城市关键词
'''

import os
import jieba
import pandas
from jieba import analyse
from washing import seg_sentence
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from scipy.misc import imread

# 城市游记清洗
for info in os.listdir(r'yj_city_content'):
    domain = os.path.abspath(r'yj_city_content')  # 获取文件夹的路径，此处其实没必要这么写，目的是为了熟悉os的文件夹操作
    info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
    inputs = open(info, "r", encoding='utf-8')
    now_city_name = inputs.name.split('\\')[-1].split('.')[-2] # inputs.name[48:-4]
    # if now_city_name != 'taiwan':
    #     continue
    # print("taiwan")
    pathc = "yj_city_content_wash_20777\\"
    if not os.path.exists(pathc):  # check if exist path
        os.makedirs(pathc)  # if not create the folder
    outputs = open(pathc + now_city_name + "_wash.txt", "w", encoding='utf-8')

    for line in inputs:
        line_seg = seg_sentence(line)
        outputs.write(line_seg + '\n')
    #
    outputs.close()
    inputs.close()

    # 城市关键词（基于词频）
    f = open(pathc + now_city_name + "_wash.txt", "r", encoding="utf-8")
    content = f.read().strip()  # 获取内容
    f.close()
    tags = jieba.analyse.extract_tags(content, topK=400)  # 采用jieba.analyse.extrack_tags(content, topK)提取关键词
    txt_freq = {}
    for i, word in enumerate(tags):
        txt_freq[word] = float(100-i*0.25)
    print(now_city_name, txt_freq)
