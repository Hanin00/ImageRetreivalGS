'''
  노드가 3개이고, 유의미한 object와 edge로 구성된 쿼리 그래프와 그 시퀀스
  
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

  total_subgraph = []
  total_subgraph_fid = []
  
  max_node = 3
  R_BFS = True
  

  
  qGraphs = []
  metadata = []
  fidList = []
  data_folder = './data/scenegraph/4239231056.json.pkl'
  with open(data_folder, "rb") as fr:
    tmp = pickle.load(fr)
    for i in range(len(tmp[0])):      
      if tmp[2][i] in [181, 190,211,214,218,220,223,232,235,238,242,261,294,296,310]:  # 0,4,7
        #[(0, 'child'), (2, 'adult'), (3, 'adult'), (4, 'adult'), (5, 'adult'), (6, 'snowboard'), (7, 'snowboard'), (8, 'snowboard')]
        #[(0, 6, 'beneath'), (0, 4, 'in_front_of'), (0, 5, 'in_front_of'), (2, 8, 'beneath'), 
        # (4, 7, 'next_to'), (4, 5, 'behind'), (4, 6, 'away'), (5, 6, 'towards'), (5, 7, 'away')]
        origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)       
        # g1의 1, 2, 3 노드와 해당 엣지를 g2로 복제
        nodes_to_clone = [0, 4, 7]
        
        qg1  = nx.Graph()
        # 노드와 노드 속성 복제
        for node in nodes_to_clone:
            print(origin_g.nodes(data=True))
            qg1.add_node(node, **origin_g.nodes[node])
        print(qg1.nodes(data=True))
          

        # 엣지 복제
        for node1 in nodes_to_clone:
            for node2 in nodes_to_clone:
                if origin_g.has_edge(node1, node2):
                    qg1.add_edge(node1, node2, **origin_g[node1][node2])
        
        qGraphs.append(qg1)
        fidList.append(tmp[2][i])
        metadata.append("4239231056")
        
        
  data_folder = './data/scenegraph/7645715544.json.pkl'
  with open(data_folder, "rb") as fr:
    tmp = pickle.load(fr)
    for i in range(len(tmp[0])):      
      if tmp[2][i] in [232, 234, 240, 244, 250, 255, 257, 302, 316]:  # 0,4,7
        #[(0, 'child'), (2, 'adult'), (3, 'adult'), (4, 'adult'), (5, 'adult'), (6, 'snowboard'), (7, 'snowboard'), (8, 'snowboard')]
        #[(0, 6, 'beneath'), (0, 4, 'in_front_of'), (0, 5, 'in_front_of'), (2, 8, 'beneath'), 
        # (4, 7, 'next_to'), (4, 5, 'behind'), (4, 6, 'away'), (5, 6, 'towards'), (5, 7, 'away')]
        origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)       
        # g1의 1, 2, 3 노드와 해당 엣지를 g2로 복제
        nodes_to_clone = [0, 2, 3]
        
        qg1  = nx.Graph()
        # 노드와 노드 속성 복제
        for node in nodes_to_clone:
            print(origin_g.nodes(data=True))
            qg1.add_node(node, **origin_g.nodes[node])
        print(qg1.nodes(data=True))
          

        # 엣지 복제
        for node1 in nodes_to_clone:
            for node2 in nodes_to_clone:
                if origin_g.has_edge(node1, node2):
                    qg1.add_edge(node1, node2, **origin_g[node1][node2])
        
        qGraphs.append(qg1)
        fidList.append(tmp[2][i])
        metadata.append("7645715544")
  
  with open(f'data/query3node/'+'seq_g3_4239231056_7645715544'+'.pkl', 'wb') as f:
      pickle.dump((qGraphs, metadata, fidList), f)
  print(  qGraphs)
  print(  metadata)
  print(  fidList)
  
  
  
  # origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[i], args, F0Dict)  # Gs에 Feature 붙임 
  #               subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)




if __name__ == "__main__":
    main()