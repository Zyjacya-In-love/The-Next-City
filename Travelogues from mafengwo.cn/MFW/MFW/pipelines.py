# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import os
import sys

import MFW.spiders.MyEncode
from MFW.citys import places

class MfwPipeline(object):
    path = "./result"
    def __init__(self):
        folder = os.path.exists(self.path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(self.path)

        # print("\n\n 现在调用了init")
    cnt = 0

    def process_item(self, item, spider):
        # print("\n\n\n\n\n", spider.place, "\n\n\n\n\n\n\n\n")
        title = json.dumps(dict(item)["title"], ensure_ascii=False, cls=MFW.spiders.MyEncode.MyEncoder) # + "\n\n\n"
        data = json.dumps(dict(item), ensure_ascii=False, cls=MFW.spiders.MyEncode.MyEncoder)  + "\n"

            # return item
        self.f = open(self.path + "//" + places[int(spider.place)] + ".json", "a",
                      encoding="utf-8")  # D:\college\Junior\python_final\MFW\MFW
        print(self.cnt, title)

        self.f.write(data)
        self.f.close()
        self.cnt += 1
        # if self.cnt > 50:
        #     print("cnt > 50")
        #     # exit()
        #     os._exit(1)
        #     # os._exit()
            # os.exit()
        return item

    # def close_spider(self, spider):
