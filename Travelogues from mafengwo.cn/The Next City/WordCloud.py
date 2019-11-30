'''
WordCloud 用于画出 城市关键词 词云
    1. 将每个城市的清洗结果依据词频得出城市关键词，再将关键词画成词云，将词云存到 yj_city_wordcloud_save 目录下
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

# 画城市词云
for info in os.listdir(r'yj_city_content'):
    domain = os.path.abspath(r'yj_city_content')  # 获取文件夹的路径，此处其实没必要这么写，目的是为了熟悉os的文件夹操作
    info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
    inputs = open(info, "r", encoding='utf-8')
    now_city_name = inputs.name.split('\\')[-1].split('.')[-2] # inputs.name[48:-4]

    pathc = "yj_city_content_wash_20777\\"
    # 城市关键词
    f = open(pathc + now_city_name + "_wash.txt", "r", encoding="utf-8")
    content = f.read().strip()  # 获取内容
    f.close()
    tags = jieba.analyse.extract_tags(content, topK=400)  # 采用jieba.analyse.extrack_tags(content, topK)提取关键词
    txt_freq = {}
    for i, word in enumerate(tags):
        txt_freq[word] = float(100-i*0.25)
    print(now_city_name, txt_freq)

    # 词云
    font_path = r'.\font\simkai.ttf'  # 为matplotlib设置中文字体路径没
    d = path.dirname(__file__)
    pathw = "yj_city_wordcloud_save\\"
    if not os.path.exists(pathw):  # check if exist path
        os.makedirs(pathw)  # if not create the folder

    img = pathw + now_city_name + "_WordCloud.png"  # 保存的图片名字

    # 设置词云属性
    wc = WordCloud(font_path=font_path,  # 设置字体
                   background_color="white",  # 背景颜色
                   max_words=400,  # 词云显示的最大词数
                   max_font_size=100,  # 字体最大值
                   random_state=42,
                   width=1000, height=860, margin=2,# 设置图片默认的大小,但是如果使用背景图片的话,那么保存的图片大小将会按照其大小保存,margin为词语边缘距离
                   )

    wc.generate_from_frequencies(txt_freq)

    plt.figure()
    # 以下代码显示图片
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
    # 绘制词云

    # 保存图片
    wc.to_file(path.join(d, img))
