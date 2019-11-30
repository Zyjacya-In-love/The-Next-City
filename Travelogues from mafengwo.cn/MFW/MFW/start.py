import os

from scrapy import cmdline
from MFW.citys import places
import subprocess
# cmdline.execute("scrapy crawl mfw".split())
for pl in places.keys():
    print(places[pl])
    # cmdline.execute(("scrapy crawl mfw" + " -a place=" + str(pl)).split())
    os.system(str("scrapy crawl mfw" + " -a place=" + str(pl) + " --nolog")) #
    # subprocess.Popen(str("scrapy crawl mfw" + " -a place=" + str(pl)))
    print("\nDone\n\n")
    # print(("scrapy crawl mfw").split())