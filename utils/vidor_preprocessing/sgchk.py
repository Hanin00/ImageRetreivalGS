import networkx as nx
import pickle
import os, sys
from tqdm import tqdm

with open('data/Vidor/scenegraph_upper6/00_upper6_merge_scenegraphs', 'rb') as fr:
  data = pickle.load(fr)

cnt = 0
for graphs in data:
  cnt += len(graphs)
print("cnt: ",cnt)



# # #fid graph 에 tag 넣어야하는데 노드에 넣을건가?
# path_to_folder = 'data/Vidor/scenegraph/'  
# file_list = os.listdir(path_to_folder)

# cnt = 0
# upper5Graph = []
# totalFileNameList = []
# for file_name in tqdm(file_list): # 파일 명 
#   with open(os.path.join(path_to_folder,file_name), "rb") as fr:
#     data = pickle.load(fr) 
#     scenegraphs = data[0]
#     for idx, videojson in enumerate(scenegraphs): #5개 파일에 대한 scenegraph list
#       video = []
#       fileNameList = []
#       for graph in videojson:
#         if len(graph) !=0 : #한개 json의 frame 한 개 = scene graph 1개
#           video.append(graph)
#           fileNameList.append(data[1][idx])
#       if len(video) != 0:
#         upper5Graph.append(video)
#         totalFileNameList.append(fileNameList)

# with open('data/Vidor/scenegraph/merge_scenegraphs', 'wb') as f:
#         pickle.dump((upper5Graph, totalFileNameList), f)

# print("len(upper5Graph) :",len(upper5Graph))
# print("len(totalFileNameList) :",len(totalFileNameList))