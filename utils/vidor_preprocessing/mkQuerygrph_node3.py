'''

  5. 특징 추출 후 전달 - edge all 모델로 추출할 것
'''
from utils import utils, subgraph
from cbir_subsg import models
from cbir_subsg.conf import parse_encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

import os, sys
import torch
import argparse
import pickle
import time
import tqdm

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt


def main():
  parser = argparse.ArgumentParser(description='Embedding arguments')
  parser.add_argument('--max_batches', type=int, default=1000, help='Maximum number of batches to train on')
  utils.parse_optimizer(parser)
  parse_encoder(parser)
  args = parser.parse_args()
  
  with open('data/class_unique_textemb.pickle', 'rb') as f:  
      data  = pickle.load(f)
  F0Dict = data

  
  qGraphs = []
  total_subgraph = []
  total_subgraph_fid = []
  
  max_node = 3
  R_BFS = True

  
  data_folder = './data/scenegraph/4239231056.json.pkl'
  with open(data_folder, "rb") as fr:
    tmp = pickle.load(fr)
    for i in range(len(tmp[0])):
      if tmp[2][i] == 214: 
        #[(0, 'child'), (2, 'adult'), (3, 'adult'), (4, 'adult'), (5, 'adult'), (6, 'snowboard'), (7, 'snowboard'), (8, 'snowboard')]
        #[(0, 6, 'beneath'), (0, 4, 'in_front_of'), (0, 5, 'in_front_of'), (2, 8, 'beneath'), 
        # (4, 7, 'next_to'), (4, 5, 'behind'), (4, 6, 'away'), (5, 6, 'towards'), (5, 7, 'away')]
        origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)       
        qg1 = origin_g.copy() # 0,4,7 - 아이, 어른(아이), 어른(아이)가 들고 있는 스노우보드/ 0번이 4,7 근처로 감      
        [qg1.remove_node(i) for i in [2,3,5,6,8]]
        
        qg2 = origin_g.copy() # 4,5,7 - 아예 붙어있는거라 좀 애매함
        [qg2.remove_node(i) for i in [0,2,3,6,8]] 
            
        qGraphs.append(qg1)
        qGraphs.append(qg2)
        
  
  data_folder = './data/scenegraph/7645715544.json.pkl'
  with open(data_folder, "rb") as fr:
    tmp = pickle.load(fr)
  for i in range(len(tmp[0])):
    if tmp[2][i] == 232: 
      #[(0, 'child'), (1, 'child'), (2, 'adult'), (3, 'adult'), (4, 'surfboard')]
      #[(0, 2, 'behind'), (0, 1, 'behind'), (0, 3, 'towards'), (1, 2, 'behind'), (1, 3, 'towards'), (2, 3, 'towards'), (3, 4, 'hold')]
      origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)       
      # print(origin_g.nodes(data="name"))
      # print(origin_g.edges(data="predicate"))
      
      qg1 = origin_g.copy() # 0,2,3 - 아이, 어른(빨간티), 어른(노란티) 0,2 옆에 있음. 3과 0,2 위치 비슷해짐
      [qg1.remove_node(i) for i in [1,4]] 
      qg2 = origin_g.copy() # 1,2,3 - 이미지 내에서 객체 전체의 위치는 변하지만, 객체 간 거리는 비슷함
      [qg2.remove_node(i) for i in [0,4]]
      
          
      qGraphs.append(qg1)
      qGraphs.append(qg2)
  
  
  data_folder = './data/scenegraph/6314288870.json.pkl'
  with open(data_folder, "rb") as fr:
    tmp = pickle.load(fr)
  for i in range(len(tmp[0])):
    if tmp[2][i] == 122: 
      # [(1, 'child'), (2, 'child'), (3, 'child'), (4, 'adult'), (5, 'baby')]
      # [(1, 2, 'behind'), (1, 3, 'behind'), (1, 4, 'next_to'), (1, 5, 'next_to'), 
      # (2, 3, 'in_front_of'), (2, 4, 'in_front_of'), (2, 5, 'in_front_of'),
      # (3, 5, 'in_front_of'), (3, 4, 'next_to'), (4, 5, 'in_front_of')]
      origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)    
      
      qg1 = origin_g.copy() # 3,4,5 - 어른 있다 없다함 -> 연속성?
      [qg1.remove_node(i) for i in [1,4]]
      qg2 = origin_g.copy() # 1,2,3 - 이미지 내에서 객체 전체의 위치는 변하지만, 객체 간 거리는 비슷함
      [qg2.remove_node(i) for i in [4,5]]
          
      qGraphs.append(qg1)
      qGraphs.append(qg2)              
  
  
  metadata = ['4239231056.json', '4239231056.json' ,
              '7645715544.json', '7645715544.json',
              '6314288870.json','6314288870.json']
  
  fidList = [214, 214, 232, 232, 122, 122]
  
  with open(f'data/query3node/'+'4239_7645_6314'+'.pkl', 'wb') as f:
      pickle.dump((qGraphs, metadata, fidList), f)
  
  
  
  # origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[i], args, F0Dict)  # Gs에 Feature 붙임 
  #               subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)




if __name__ == "__main__":
    main()