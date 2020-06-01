# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:41:51 2020

@author: pc
"""


import jieba 
import snownlp
import thulac
import pkuseg
import jiagu

import fool

from pyhanlp import HanLP


import time


sample_1 = "小明一把把把把住了"
sample_2 = "小白痴痴地等着她回来"
sample_3 = "汽水不如果汁好喝, 买水果然后榨汁"
sample_4 = "吃完饭和尚未吃饭的人都在等着外卖小哥的到来"

samples = [sample_1, sample_2, sample_3, sample_4]

start_time = time.time()
for sample in samples:
    print(jieba.lcut(sample))
end_time = time.time()
print('jieba time', (end_time - start_time) * 1000)

start_time = time.time()
for sample in samples:
    print(snownlp.SnowNLP(sample).words)
end_time = time.time()
print('snownlp time', (end_time - start_time) * 1000)

start_time = time.time()
thul = thulac.thulac(seg_only=True)
for sample in samples:
    print(thul.cut(sample, text=True))
end_time = time.time()
print('thulac time', (end_time - start_time) * 1000)

start_time = time.time()
seg = pkuseg.pkuseg()          
for sample in samples:
    print(seg.cut(sample))
end_time = time.time()
print('pkuseg time', (end_time - start_time) * 1000)

start_time = time.time()
for sample in samples:
    print(jiagu.seg(sample))
end_time = time.time()
print('jiagu time', (end_time - start_time) * 1000)

start_time = time.time()
for sample in samples:
    print(fool.cut(sample))
end_time = time.time()
print('fool time', (end_time - start_time) * 1000)

start_time = time.time()
for sample in samples:
    print(HanLP.segment(sample))
end_time = time.time()
print('HanLP time', (end_time - start_time) * 1000)




