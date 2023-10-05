import networkx as nx
import pickle
import os, sys
from tqdm import tqdm



import os
import shutil

# 원본 폴더와 목표 폴더 경로를 설정합니다.
source_folder = 'data/training_annotation/'
target_folder = 'data/training_total/'

# 원본 폴더 안의 모든 파일과 하위 폴더를 검사합니다.
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.endswith('.json'):
            # 확장자가 .json인 경우
            source_path = os.path.join(root, filename)
            target_path = os.path.join(target_folder, filename)

            # 파일을 목표 폴더로 이동합니다.
            shutil.move(source_path, target_path)

print("모든 파일을 이동했습니다.")

sys.exit()



# with open('data/Vidor/scenegraph/merge_scenegraphs_1.pkl', 'rb') as fr:
#   data = pickle.load(fr)
# print(len(data[0][0]))
# print(data[0][0][0])


# sys.exit()




# with open('data/Vidor/scenegraph/0_2754378442_6188920051.pkl', 'rb') as fr:
#   data = pickle.load(fr)
# print(data[0][0])
# print(len(data[0]))
# print(len(data[0][0]))

# for video in data[0]:
#   for graph in video:
#     if len(graph) != 0:
#       print(len(graph))

# ------------------------------
# with open('data/Vidor/scenegraph/merge_scenegraphs.pkl', 'rb') as fr:
#   data = pickle.load(fr)

# cnt = 0
# for graphs in tqdm(data[0]) :
#   cnt += len(graphs)
# print("cnt: ",cnt)


# sys.exit()
# ------------------------------
# with open('data/Vidor/scenegraph/merge_scenegraphs', 'rb') as fr:
#   data = pickle.load(fr)

# cnt = 0
# for graphs in tqdm(data[0]):
#   cnt += len(graphs)
# print("cnt: ",cnt)


# # #fid graph 에 tag 넣어야하는데 노드에 넣을건가?
path_to_folder = 'data/scenegraph/'  
file_list = os.listdir(path_to_folder)

listIdx = len(file_list)//4

file_list1 = file_list[:listIdx]
file_list2 = file_list[listIdx : listIdx*2]
file_list3 = file_list[listIdx*2 : listIdx*3]
file_list4 = file_list[listIdx*3 :]
target_list = [file_list1, file_list2, file_list3, file_list4]


#비어있는 scenegraph filtering
for idx in range(0, len(file_list), 5): # 파일 명 
  for idx2 in range(5):
    try:
      file_list_name = file_list[idx+idx2]
    except:
      file_list_name = file_list[:-1]
    cnt = 0
    upper5Graph = []
    totalFileNameList = []
    with open(os.path.join(path_to_folder,file_list_name), "rb") as fr:
      data = pickle.load(fr) 
      scenegraphs = data[0]
      for idx2, videojson in enumerate(scenegraphs): #5개 파일에 대한 scenegraph list
        video = []
        fileNameList = []
        for graph in videojson:
          if len(graph) !=0 : #한개 json의 frame 한 개 = scene graph 1개
            video.append(graph)
            fileNameList.append(data[1][idx2])
        if len(video) != 0:
          cnt += len(video)
          upper5Graph.append(video)
          totalFileNameList.append(fileNameList)

  with open('data/scenegraph_merge/merge_scenegraphs_{}.pkl'.format(idx), 'wb') as f:
          pickle.dump((upper5Graph, totalFileNameList), f)

  print("cnt: ", cnt)
  print("len(upper5Graph) :",len(upper5Graph))
  print("len(totalFileNameList) :",len(totalFileNameList))

