'''
Json2TxT 用于整理爬虫的结果
    1. 将每个城市的所有游记具体内容集合在一起，每个城市保存为一个 .txt 文件，并存到 yj_city_content 目录下
    2. 将所有城市的游记内容集合在一起，存到 yj_all_content 目录下
'''

import os
import json

# 将每个城市的所有游记具体内容集合在一起，并存到yj_city_content下
for info in os.listdir(r'D:\college\Junior\AI_final\MFW\MFW\result'):
    domain = os.path.abspath(r'D:\college\Junior\AI_final\MFW\MFW\result')  # 获取文件夹的路径
    info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
    input_f = open(info, 'r', encoding='utf-8')
    fcontent = input_f.read().split('\n')
    del fcontent[len(fcontent) - 1]

    yj_city_content = ""
    # print(input_f.name.split('\\')[-1].split('.')[-2])
    for i in range(len(fcontent)):
        if i >= 75:
            break
        jsonobject = json.loads(fcontent[i])
        yj_city_content += jsonobject["content"] + "\n"

    # print(input_f.name.split('\\')[-1].split('.')[-2], " ", len(fcontent))
    path = ".\\yj_city_content"
    if not os.path.exists(path):  # check if exist path
        os.makedirs(path)  # if not create the folder
    output_f = open(path + "\\" + input_f.name.split('\\')[-1].split('.')[-2] + ".txt", 'w', encoding='utf-8')
    output_f.write(yj_city_content)

    input_f.close()
    output_f.close()


# # 将全部城市的所有游记具体内容集合在一起，并存到yj_all_content下
# all_city_content = ""
# for info in os.listdir(path):
#     domain = os.path.abspath(path)
#     info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
#     input_f = open(info, 'r', encoding='utf-8')
#     all_city_content += input_f.read()
#     input_f.close()
#
# path = ".\\yj_all_content"
# if not os.path.exists(path):  # check if exist path
#     os.makedirs(path)  # if not create the folder
# output_f = open(path + "\\allcontent.txt", 'w', encoding='utf-8')
# output_f.write(all_city_content)
# output_f.close()
