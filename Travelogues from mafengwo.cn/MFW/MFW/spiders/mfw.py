# -*- coding: utf-8 -*-
import scrapy
from MFW.items import MfwItem


class MfwSpider(scrapy.Spider):
    name = 'mfw'
    allowed_domains = ['mafengwo.cn']
    baseURL = "http://www.mafengwo.cn/yj/"
    middleURL = "/1-0-"
    endURL = ".html"
    place = 0
    offset = 1
    def __init__(self, place, *args, **kwargs):
        super(MfwSpider, self).__init__(*args, **kwargs)
        self.place = place
        self.start_urls = [self.baseURL + str(place) + self.middleURL + str(self.offset) + self.endURL]

    # def start_requests(self):
    #     for pl in self.places:
    #         self.place = pl
    #         url = self.baseURL + str(self.place) + self.middleURL + str(self.offset) + self.endURL
    #         yield self.make_requests_from_url(url)

    def parse(self, response):
        nodelist2 = response.xpath("//li[@class='post-item clearfix']")
        # print("---------------------------------------nodelist2 : ", len(nodelist2),"---------------------------------------------------")
        for node2 in nodelist2:

            a = node2.xpath("./h2/a[@target = '_blank']/@href").extract()
            # print("This is a !!!\n" , a)
            if len(a) == 2:
                conlink = a[1]
            else:
                conlink = a[0]
            yield scrapy.Request('http://www.mafengwo.cn'+conlink, callback=self.parse_item)

        if self.offset < 10:
            self.offset += 1
            url = self.baseURL + str(self.place) + self.middleURL + str(self.offset) + self.endURL
            # print("this is url \n", url, "\n");
            yield scrapy.Request(url, callback=self.parse)
    cnt = 0
    def parse_item(self, response):
        item = MfwItem()
        self.cnt += 1
        # print("cnt : ", self.cnt)
        item["destination"] = response.xpath("//a[@class = '_j_mdd_stas special_mdd_ico']/@title").extract()[0].encode("utf-8")
        item["author"] = response.xpath("//meta[@name = 'author']/@content").extract()[0].encode("utf-8")

        if len(response.xpath("//li[@class = 'time']/text()[2]")):
            item["time"] = response.xpath("//li[@class = 'time']/text()[2]").extract()[0].encode("utf-8")
        else:
            item["time"] = ""

        if len(response.xpath("//h1[@class = 'headtext lh80']/text()")):
            item["title"] = response.xpath("//h1[@class = 'headtext lh80']/text()").extract()[0].encode("utf-8")
        else:
            item["title"] = ""

        data = response.selector.xpath("//div[@class = 'vc_article']")
        item["content"] = data.xpath('normalize-space(string(.))').extract()[0].encode("utf-8")
        # content = data.xpath('normalize-space(string(.))').extract()[0]
        # highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        # item["content"] = highpoints.sub('--emoji--', content)
        yield item


