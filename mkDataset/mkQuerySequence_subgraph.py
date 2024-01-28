'''
만든이: haeun5419@a.ut.ac.kr
코드 개요: Query를 위한 Query graph 생성


1. 서브 그래프 생성
2. 노드 또는 엣지의 추가/제거
! 이때 노드는 이미 생성된 서브 그래프들과 걸쳐있되, 동일하진 않아야함
    -> 해당 내용을 살펴보려면 그래프가 충분히 커야할 것으로 보임



두 개의 쿼리 비디오를 선정하고, 거기에서 node를 제거해서 사용하는 것이 나아보임.
사용 비디오: 3802296828

'''
import sys, os
import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import pickle

from utils import utils, subgraph

# import math
# # 이미지의 가로와 세로 길이
# width = 1980
# height = 2140

# # 이미지의 대각선 길이 계산
# diagonal_length = math.sqrt(width**2 + height**2)
# diagonal_length

# def normalize_distance(distance, max_distance):
#     return distance / max_distance

# # 예제 거리 (500픽셀) 사용
# example_distance = 500
# normalized_distance = normalize_distance(example_distance, diagonal_length)
 

# 서브 그래프 생성 후 이상 행동에서 scenegraph 생성 - 

querys = []
db = []
db_idx = []

max_node = 3
R_BFS = True

with open("data/retrieval_test_subgraph/subgraph_dataset_8402409351.pkl", "rb") as fr:
        db, db_idx, db_reIdx = pickle.load(fr)  
for  i in range(len(db)):
    if len(db[i].nodes()) > 5:
        print(db[i].nodes(data="name"))

sys.exit()







# 근데 쿼리 그래프는 rpe가 없는디.. 어카지.. 이게 영향을 미치진 않나..? -> 


# vId = str(4239231056)
# vId = str(8888541668)

#한 장면에서 10개 이상의 객체가 나오는 비디오

# vId = str(8402409351)
# vId = str(8718578847)
# vId = str(6121794107)
# vId = str(6514779107)
# vId = str(12292269854)
# vId = str(3881536171)


# folder_path = 'data/dataset02/scenegraph_v2/'

# # 폴더 내의 모든 파일 및 디렉토리 목록을 가져옴
# file_list = os.listdir(folder_path)
# for vId in file_list:    
#     with open("data/dataset02/scenegraph_v2/"+vId, "rb") as fr:
#         db, db_idx, db_reIdx = pickle.load(fr)  

    
#     for  i in range(len(db)):
#         if len(db[i].nodes()) > 10:
#             print("vId : ",vId)
#             print(db[i].nodes(data="name"))
            

# sys.exit()



with open("data/dataset02/retrieval/retrieval_db.pkl", "rb") as fr:
        db, db_idx, db_reIdx, cnt_video = pickle.load(fr)  
for  i in range(len(db)):
    if len(db[i].nodes()) > 5:
        print(db[i].nodes(data="name"))

sys.exit()







# print(db[0].nodes(data=True))
print(db[0].nodes(data="name"))
# print(db[0].edges(data=True))


print(db[0].edges(data="predicate"))

print(len(db[0].nodes()))
print(len(db[0].edges()))

sys.exit()

print(db[0])
print(db[0].nodes(data=True))
print(db[0].edges(data=True))
sys.exit()

#2. subgraph에서 노드 삭제



#3. 




print(len(db_idx[0])) #548
print(db_idx[0][0])


sys.exit()




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






