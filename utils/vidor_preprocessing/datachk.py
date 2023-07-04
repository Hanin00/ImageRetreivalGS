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
  

    

def use1video(data, use1video):
  gList = []
#scene graph  
  for idx, img in enumerate(data['trajectories']):
    G = mk1Graph(img, use1video) # json 하나에서 trajectories 하나를 기준으로 graph 하나 만듦 -> scene graph 한장 당 프레임id 하나에 매칭됨
    # print(G.nodes(data=True))
    gList.append(G)
    # if len(gList) == 395: 
    #    print(gList[394])
    #    sys.exit()
  return gList
   

   

# json 하나에서 trajectories 하나를 기준으로 graph 하나 만듦 -> scene graph 한장 당 프레임id 하나에 매칭됨
def mk1Graph(sceneG, txtemb):
  testG = nx.Graph()
  for idx in range(len(sceneG)):  
    if len(sceneG[idx]['bbox']) > 0 :
      testG.add_nodes_from([(sceneG[idx]['tid'], {'bbox': sceneG[idx]['bbox'], 
                                            'generated': sceneG[idx]['generated'],
                                            'tracker': sceneG[idx]['tracker']})])
  return testG

#relation_instance 기준으로 predicate 및 edge 생성, bbox 기준으로 attribute 추가
def addEdge(gList,relation): 
   for rel in relation:
      for idx in (rel['begin_fid'], rel['end_fid'] ):
        #  print("idx: ",idx)
         for idx, g in enumerate(gList):
          g.add_edges_from([(rel['subject_tid'],rel['object_tid'], {'predicate': rel['predicate']})])
        
          # 객체 A와 B의 bbox 정보 추출
          try: 
            bbox_a = g.nodes[rel['subject_tid']]['bbox']
            # print(g.nodes[rel['subject_tid']])

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
                                (rel['subject_tid'],rel['object_tid'], {'angle_BA': angle_BA})])
          except:
             continue
          #    print("idx: ",idx)
          #    print(g.nodes(data=True))
          #    print(g.nodes[rel['subject_tid']])
          #    print([rel['subject_tid']])
          #    sys.exit()


  



def main():
  # mkTextEmb()

  with open("data/Vidor/class_unique_textemb.pickle", "rb") as fr:
    synsDict = pickle.load(fr)



  totalGraphs = []
  # # 폴더 경로 지정
  path_to_folder = 'data/Vidor/training_total/'
  # mkTotalClass(path_to_folder)
  folder_list = os.listdir(path_to_folder) 
  for folder in tqdm(folder_list):
    file_list = os.listdir(os.path.join(path_to_folder,folder))
    for json_file in file_list:
        file_path = os.path.join(path_to_folder, folder, json_file)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            gList = use1video(json_data, synsDict) #1 json data = 1 video data
            addEdge(gList, json_data['relation_instances'])  #relation_instance 기준으로 predicate 및 edge 생성, bbox 기준으로 attribute 추가
            totalGraphs.extend(gList)
    print("len(totalGraphs): ", len(totalGraphs))


            
            
            

    
if __name__ == "__main__":
   main()