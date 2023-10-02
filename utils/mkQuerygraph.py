'''
만든이: haeun5419@a.ut.ac.kr
코드 개요: Query를 위한 Query graph 생성

두 개의 쿼리 비디오를 선정하고, 거기에서 node를 제거해서 사용하는 것이 나아보임.
사용 비디오: 3802296828

'''
import sys
import numpy as np
import pandas as pd
import torch
import csv


import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import pickle


querys = []
with open('data/scenegraph/3802296828.json.pkl', 'rb') as f:
    scenegraphs = pickle.load(f)
print(len(scenegraphs[0])) #548


queryG1origin =scenegraphs[0][0]
queryG1origin.remove_node(0)
querys.append(queryG1origin)


queryG1origin =scenegraphs[0][324]
queryG1origin.remove_node(12)
querys.append(queryG1origin)


queryG1origin =scenegraphs[0][-200]
queryG1origin.remove_node(2)
querys.append(queryG1origin)


queryG1origin =scenegraphs[0][-134]
queryG1origin.remove_node(2)
queryG1origin.remove_node(4)
querys.append(queryG1origin)


queryG1origin =scenegraphs[0][-112]
queryG1origin.remove_node(1)
queryG1origin.remove_node(2)
queryG1origin.remove_node(4)
querys.append(queryG1origin)


# 5
queryG1origin =scenegraphs[0][-20]
queryG1origin.remove_node(5)
queryG1origin.remove_node(11)
querys.append(queryG1origin)



with open('data/scenegraph/6673828083.json.pkl', 'rb') as f:
    scenegraphs = pickle.load(f)
print(len(scenegraphs[0])) #779


queryG1origin =scenegraphs[0][0]
queryG1origin.remove_node(2)
queryG1origin.remove_node(4)
querys.append(queryG1origin)

queryG1origin =scenegraphs[0][300]
queryG1origin.remove_node(1)
queryG1origin.remove_node(4)
querys.append(queryG1origin)

queryG1origin =scenegraphs[0][450]
queryG1origin.remove_node(1)
queryG1origin.remove_node(4)
querys.append(queryG1origin)

queryG1origin =scenegraphs[0][451]
queryG1origin.remove_node(1)
queryG1origin.remove_node(7)
querys.append(queryG1origin)

queryG1origin =scenegraphs[0][532]
queryG1origin.remove_node(9)
queryG1origin.remove_node(1)
queryG1origin.remove_node(0)
querys.append(queryG1origin)

print(len(querys))
print(len(querys[0]))
[print(i.nodes(data="name")) for i in querys]

with open("data/query_graphs.pkl", "wb") as fw:
    pickle.dump(querys , fw)

with open('data/query_graphs.pkl', 'rb') as f:
    scenegraphs = pickle.load(f)
[print(i.nodes(data="name")) for i in scenegraphs]
[print(i.edges(data="distance")) for i in scenegraphs]




sys.exit()





