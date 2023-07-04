'''
  Vidor 데이터셋의 training_annotation을 기반으로 scene graph를 생성
  1. subject/object의 tid : category를 기반으로 node를 생성 및 
  전체 비디오의 subject/object의 category를 fast text로 embedding 해 10dim feature를 생성함
  - subject/object 의 tid : category는 매 비디오마다 달라지며, category 는 80개이다. 
  2. trajectories를 기준으로 tid 의 bbox를 비롯한 attribute를 생성
  3. relationship은 begin_fid, end_fid로 node(tid)간의 relation을 표현한다. 
  4. 생성한 scene graph 는 /data/Vidor/scenegraph 에 각 jsonfile(1개의 video)와 동명인 pickle file로 저장된다. 



'''


import os, sys
import numpy as np
import pandas as pd
import math
import torch
import csv
import torch_geometric.utils

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import FastText
import fasttext.util
from tqdm import tqdm
import time
import json
from collections import Counter
import pickle
import nltk
from nltk.corpus import conll2000

# https://daydreamx.tistory.com/entry/NLP-FastText-%EC%9D%98-pretrained-model%EC%97%90-text-classification%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%B6%94%EA%B0%80%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0 - fasttext 사용




'''
  전체 데이터 셋을 대상으로 해야하는 것들
  1. text embedding을 위한 class 수집 -> 일단 전체 subject/objects
  2. 

  Guess1. 영상 사이즈도 고려해서 동일하게 변경해야하는 것 아닌가? bbox를 특징으로 할거면 regulize도 필요한 것 아닌가?
'''

# dict_keys(['version', 'video_id', 'video_hash', 'video_path',
#  'frame_count', 'fps', 'width', 'height', 'subject/objects', 
#  'trajectories', 'relation_instances'])

#for Gid in range(totalG) : #Graph level 
# for relation in range(data['relation_instances']): #relation level <- Video 에서 fid 로 구간 

def mkTotalClass(path_to_folder): # 걸린 시간 : 79.74567 sec
    classList = []
    # # 폴더 경로 지정
    # path_to_folder = 'data/Vidor/training_annotation/'
    folder_list = os.listdir(path_to_folder) 
    
    for folder in tqdm(folder_list):
      file_list = os.listdir(os.path.join(path_to_folder,folder))
      for json_file in file_list:
          file_path = os.path.join(path_to_folder, folder, json_file)
          with open(file_path, 'r') as f:
              json_data = json.load(f)
              [classList.append(json_data['subject/objects'][idx]['category']) for idx in range(len(json_data['subject/objects']))]

    # 데이터프레임 생성
    
    df = pd.DataFrame({'classList': classList})
    # CSV 파일로 저장
    df.to_csv('data/Vidor/classList.csv', index=False)

    unique_classList =  list(set(classList))
    df = pd.DataFrame({'classList': unique_classList})
    # CSV 파일로 저장
    df.to_csv('data/Vidor/classList_unique.csv', index=False)




def mkTextEmb():
  corpus_fname = pd.read_csv('data/Vidor/classList_unique.csv')
  model_fname = '/home/dblab/Haeun/CBIR/ImageRetreivalGS/data/Vidor/model'
  words = corpus_fname['classList'].tolist()

  model = fasttext.load_model('cc.en.300.bin')
  fasttext.util.reduce_model(model, 10)
  # model = FastText(words, size=10, workers=4, sg=1, iter=6, word_ngrams=1)
  # model.save(model_fname)

  synsDict = {}
  for idx, word in enumerate(words):
    synsDict.update({word : model.get_word_vector(word)})
    # print(word,": " ,"emb: " , model.get_word_vector(word))
    # print(len(model.get_word_vector(word)))
    # sys.exit()

  with open("data/Vidor/class_unique_textemb.pickle", "wb") as fw:
    pickle.dump(synsDict, fw)
  

    
'''
   tid_category_dict - 비디오마다 계속 바뀜 - tid: category
   synsDict - 전체 데이터셋 대상으로 생성 - 80개 category - class: embedding feature
'''
def use1video(data, synsDict):
  gList = []
# subj/obj id - class dict
  tid_category_dict = {
      item['tid']: item['category'] for item in data['subject/objects']
  }
#scene graph    
  for idx, img in enumerate(data['trajectories']):
    # print("img: ",img)
    if len(img)>5:
      # print("img: ",img)
      # print("len(img): ",len(img))
      G = mk1Graph(img, synsDict, tid_category_dict) # json 하나에서 trajectories 하나를 기준으로 graph 하나 만듦 -> scene graph 한장 당 프레임id 하나에 매칭됨 
      gList.append(G)
      # if len(gList) == 395: 
      #   print(gList[394])
      #   sys.exit()
  return gList
  

# json 하나에서 trajectories 하나를 기준으로 graph 하나 만듦 -> scene graph 한장 당 프레임id 하나에 매칭됨
def mk1Graph(sceneG, synsDict, tid_category_dict):
  testG = nx.Graph()
  for idx in range(len(sceneG)):  
    if len(sceneG[idx]['bbox']) > 0 :
      testG.add_nodes_from([(sceneG[idx]['tid'], {'bbox': sceneG[idx]['bbox'], 
                                            'generated': sceneG[idx]['generated'],
                                            'tracker': sceneG[idx]['tracker'],
                                            'txtemb': synsDict[tid_category_dict[sceneG[idx]['tid']]],
                                            'name': tid_category_dict[sceneG[idx]['tid']],
                                            })])
  return testG

#relation_instance 기준으로 predicate 및 edge 생성, bbox 기준으로 attribute 추가
def addEdge(gList,relation): 
   cnt = 0
   for rel in relation:
      for idx in (rel['begin_fid'], rel['end_fid'] ):
        #  print("idx: ",idx)
         for idx, g in enumerate(gList):
          # 객체 A와 B의 bbox 정보 추출
          try: 
            bbox_a = g.nodes[rel['subject_tid']]['bbox']
            bbox_b = g.nodes[rel['object_tid']]['bbox']
            # A와 B의 중심 좌표 계산
            center_a = ((bbox_a['xmin'] + bbox_a['xmax']) / 2, (bbox_a['ymin'] + bbox_a['ymax']) / 2)
            center_b = ((bbox_b['xmin'] + bbox_b['xmax']) / 2, (bbox_b['ymin'] + bbox_b['ymax']) / 2)
            # A와 B의 거리 계산
            distance = math.sqrt((center_b[0] - center_a[0])**2 + (center_b[1] - center_a[1])**2)
            # 객체 A를 기준으로 객체 B의 상대 각도 계산
            deltaX_AB = center_b[0] - center_a[0]
            deltaY_AB = center_b[1] - center_a[1]
            angle_AB = math.degrees(math.atan2(deltaY_AB, deltaX_AB))
            # 객체 B를 기준으로 객체 A의 상대 각도 계산
            deltaX_BA = center_a[0] - center_b[0]
            deltaY_BA = center_a[1] - center_b[1]
            angle_BA = math.degrees(math.atan2(deltaY_BA, deltaX_BA))

            g.add_edges_from([(rel['subject_tid'],rel['object_tid'], {'distribute': distance}),
                              (rel['subject_tid'],rel['object_tid'], {'angle_AB': angle_AB}),
                                (rel['subject_tid'],rel['object_tid'], {'angle_BA': angle_BA}),
                                (rel['subject_tid'],rel['object_tid'], {'predicate': rel['predicate']})
                                ])
            
          except:
             continue
          #    print("idx: ",idx)
          #    print(g.nodes(data=True))
          #    print(g.nodes[rel['subject_tid']])
          #    print([rel['subject_tid']])
          #    sys.exit()


# 데이터셋의 trajectory에 아예 없는 frame이 있어 해당 프레임을 삭제함
def dropEmpty(gList):
   dropedList = [g for g in gList if len(g.nodes) > 1]
  #  print(len(dropedList))
   dropedList = [g for g in gList if len(g.edges) > 0]
  #  print("len(dropedList) - ", len(dropedList))
   dropedList = [g for g in gList if len(g.nodes) > 1]
  #  print("node num")
  #  print([len(g.nodes()) for g in gList])

   return dropedList


def main():
  # mkTextEmb()
  with open("data/Vidor/class_unique_textemb.pickle", "rb") as fr:
    synsDict = pickle.load(fr)

  # # 폴더 경로 지정
  path_to_folder = 'data/Vidor/training_total/'

# 약 100만개의 이미지에 대해 추출하기 위해 사용
  file_path = 'data/Vidor/out.txt'
  file_list = []
  with open(file_path, "r") as file:
     for line in file:
        file_name = line.split(":")[0] # :로 strip 후 앞부분만 추출
        file_list.append(file_name)
  
  totalGList = []
  for json_file in file_list:
    file_path = os.path.join(path_to_folder, json_file)
    with open(file_path, 'r') as f:
        # print("file_name: " ,json_file)
        json_data = json.load(f)
        # use1video(json_data, synsDict) #1 json data = 1 video data
        gList = use1video(json_data, synsDict) #1 json data = 1 video data
        # print("before drop empty:", len(gList))
        addEdge(gList, json_data['relation_instances'])  #relation_instance 기준으로 predicate 및 edge 생성, bbox 기준으로 attribute 추가
        # totalGraphs.extend(gList)
        gList = dropEmpty(gList)
        # print("After drop empty:", len(gList))
        totalGList += gList
    # with open("data/Vidor/scenegraph/{}.pickle".format(json_file[:-5]), "wb") as fw:
    #   pickle.dump(gList, fw)

  with open("data/Vidor/scenegraph/total_glist_out.pickle", "wb") as fw:
      pickle.dump(totalGList, fw) 
  
  print("len(totalGList): ",len(totalGList))


            

    
if __name__ == "__main__":
   main()