'''
  1. 기존에 만든 scenegraph 불러옴
  2. subgraph의 mkSubgraph 로 서브그래프 생성함(노드 크기 4개인 것 저장, 아니면 버림)
  3. 100만 개 될 때까지 반복
  4. 저장
  
  5. 특징 추출 후 전달
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
  
  with open('data/fileNameList_ordered.pkl', 'rb') as f:
      fileNameList  = pickle.load(f)
  filenames = []
  for item in fileNameList:
      # sliced_item = item[0:3]
      sliced_item = item[-5:-4]
      filenames.extend(sliced_item)

  filenames = filenames[:1]    
  
  total_subgraph = []
  total_subgraph_fid = []
  
  
  max_node = 3
  R_BFS = True

  data_folder = 'data/scenegraph/'
  for foldername in os.listdir(data_folder):
      file_names = os.listdir(os.path.join(data_folder, foldername))
      
      with open(file_names, "rb") as fr:
        tmp = pickle.load(fr)
        for i in range(len(tmp[0])):
            origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)
            tmpsub = subgraph.make_subgraph(tmp[0][i], max_node, False, R_BFS)
            for subg in tmpsub:
              if len(subg.nodes()) == 3:
                total_subgraph.append(subg)
                total_subgraph_fid.append(tmp[2][i])
  print(total_subgraph)
  print(total_subgraph_fid)
  sys.exit()
  
                
                
            
            
            





  # origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[i], args, F0Dict)  # Gs에 Feature 붙임 
  #               subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)




if __name__ == "__main__":
    main()