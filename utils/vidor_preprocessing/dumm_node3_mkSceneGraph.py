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
  # argument
  parser = argparse.ArgumentParser(description='Embedding arguments')
  parser.add_argument('--max_batches', type=int, default=1000, help='Maximum number of batches to train on')
  utils.parse_optimizer(parser)
  parse_encoder(parser)
  args = parser.parse_args()


  # node txtemb dictionary 
  with open('data/class_unique_textemb.pickle', 'rb') as f:  
      data  = pickle.load(f)
  F0Dict = data
  
  # file size ordering
  with open('data/fileNameList_ordered.pkl', 'rb') as f:
      fileNameList  = pickle.load(f)
  filenames = []
  for item in fileNameList:
      # sliced_item = item[0:3]
      sliced_item = item[-5:-4]
      filenames.extend(sliced_item)

  filenames = filenames[:3]    
  total_subgraph = []
  total_subgraph_fid = []
  
  max_node = 3
  R_BFS = True

  # load scenegraph -> calculate & assign rpe -> make subgraphs
  data_folder = 'data/scenegraph/'
  # for foldername in os.listdir(data_folder):
  for filename in filenames:
    file_name = os.path.join(data_folder, filename)
    with open(file_name, "rb") as fr:
      tmp = pickle.load(fr)
      for i in range(len(tmp[0])):
          origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)
          tmpsub = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)
          for subg in tmpsub:
            if len(subg.nodes()) == 3:
              total_subgraph.append(subg)
              total_subgraph_fid.append(tmp[2][i])
  with open("data/dummdata/only_3nodesGraph-.pkl", "wb") as fw:
      pickle.dump([total_subgraph,total_subgraph_fid], fw)
  with open('data/dummdata/only_3nodesGraph-.pkl', 'rb') as f:
      list2 = pickle.load(f)

  
  # model load -> extract embedding features
  if not os.path.exists(os.path.dirname(args.model_path)):
      os.makedirs(os.path.dirname(args.model_path))
  model = models.GnnEmbedder(args.feature_dim, args.hidden_dim, args)
  model.to(utils.get_device())
  if args.model_path:
      model.load_state_dict(torch.load(
          args.model_path, map_location=utils.get_device()))
  else:
      return print("model does not exist")

  model.eval()
  model.zero_grad()
  
  db_data = utils.batch_nx_graphs_rpe(total_subgraph, None)
  emb_db_data = model.emb_model(db_data)

  with torch.no_grad():
      emb_db_data = model.emb_model(db_data)
      # print("emb_db_data: ", emb_db_data)
      fIdList.extend([fId] * len(subs))
      subGFeature.extend(emb_db_data) #feature
      vIdList.extend([vId] * len(subs))

  print("len(fIdList) : ",len(fIdList))
  print("len(subGFeature) : ",len(subGFeature))
  print("len(vIdList) : ",len(vIdList))
  
  subGFeature_numpy = [item.detach().cpu().numpy() for item in subGFeature]

  df = pd.DataFrame({"vId": vIdList, "fId": total_subgraph_fid,"subGFeature": subGFeature_numpy,})
  
  df.to_parquet("subgraphFeature_GAT_edgeattrall_0-3.parquet", engine="pyarrow", compression="gzip")
  # df.to_parquet('subgraphFeature_cbir.parquet', engine='fastparquet', compression='snappy')
  df.to_csv("subgraphFeature_GAT_edgeattrall_0-3.csv")  
  
  
            
            
            
            



if __name__ == "__main__":
    main()