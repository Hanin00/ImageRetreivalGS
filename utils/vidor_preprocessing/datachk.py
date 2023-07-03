import sys
import numpy as np
import pandas as pd
import math
import torch
import csv
import torch_geometric.utils

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import FastText
from tqdm import tqdm
import time
import json
from collections import Counter
import pickle
import nltk
from nltk.corpus import conll2000






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

# def mkTextEmb(totalClass):


def mk1Graph(sceneG):
  testG = nx.Graph()
  for idx in range(len(sceneG)):  
    testG.add_nodes_from([(sceneG[idx]['tid'], {'bbox': sceneG[idx]['bbox'], 
                                        'generated': sceneG[idx]['generated'],
                                        'tracker': sceneG[idx]['tracker']})])
  return testG

def addEdge(gList,relation):
   for rel in relation:
      print("rel:", rel)
      for idx in (rel['begin_fid'], rel['end_fid'] ):
         print("idx: ",idx)
         for g in gList:
          g.add_edges_from([(rel['subject_tid'],rel['object_tid'], {'predicate': rel['predicate']})])

          # 객체 A와 B의 bbox 정보 추출
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

          
          # # A와 B의 상대적인 각도 계산 (아크 탄젠트 사용)
          # angle_a = math.atan2(center_b[1] - center_a[1], center_b[0] - center_a[0])
          # angle_b = math.atan2(center_a[1] - center_b[1], center_a[0] - center_b[0])

          print("Distance:", distance)
          print("Angle of Object 0:", math.degrees(angle_AB))
          print("Angle of Object 1:", math.degrees(angle_BA))


          g.add_edges_from([(rel['subject_tid'],rel['object_tid'], {'distribute': distance}),
                             (rel['subject_tid'],rel['object_tid'], {'angle_AB': angle_AB}),
                             (rel['subject_tid'],rel['object_tid'], {'angle_BA': angle_BA})])
                              
          
          
         print(g.edges(data=True))
         sys.exit()
        #  gList[idx]



  
def main():
  gList = []
  imgCnt = 1000
  start = time.time()

  with open('data/Vidor/training_annotation/1106/3038245970.json') as file:  # open json file
      data = json.load(file)
  end = time.time()

  # 파일 읽는데 걸리는 시간 : 24.51298 sec
  print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec")
  #scene graph
  jsonG = data['trajectories'][0]
  G = mk1Graph(jsonG)
  print(G.nodes(data=True))
  gList.append(G)
  print(gList)
  print(gList[0].nodes(data=True))
  addEdge(gList, data['relation_instances'])

  # print(G.nodes(data=True))

  #   # text embedding 값 추가
  #   # todo bbox 기반의 edge 생성

  # #todo frame id 기준 graph edge 추가
  # print(G.edges(data=True))




  


    
if __name__ == "__main__":
   main()