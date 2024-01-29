from utils import utils, subgraph
# from cbir_subsg.conf_all import parse_encoder
from cbir_subsg.conf import parse_encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import argparse

import os, sys
import torch
import pickle


import numpy as np
import networkx as nx
import matplotlib.pyplot as pl


#retrieval_subgraph에서 
def mkOverlappedQuery(args,F0Dict):
    # with open("data/scenegraph_1/0_6096540713_6096540713.pkl", "rb") as fr:
    #     datas = pickle.load(fr)
    db = []
    db_idx = []
    
    seeds = 4
    
    max_node = 3
    R_BFS = True
    # 2430799380 빼고 해보기..?
    # filenames = ['4239231056.json.pkl', '7645715544.json.pkl','2406271188.json.pkl']
    
    # filenames = ['4239231056.json.pkl', '7645715544.json.pkl'] #2
    
    filenames = ['4239231056.json.pkl', '7645715544.json.pkl', '2406271188.json.pkl', '2430799380.json.pkl', 
    #  ] # 4
                 '3828379201.json.pkl', '4148862873.json.pkl', '5454696393.json.pkl', '5759653927.json.pkl'] #1215
    
    cnt_video = 0
    for filename in filenames:
        vId = filename.split('.')[0]
        
        with open("data/scenegraph/"+ filename, "rb") as fr:
            tmp = pickle.load(fr)            
            length = len(tmp[0]) 
            print("length : ",length)
        
            cnt_video += length
            # length = 2            
            if length != 0:
                cnt = 0
                # 'rpe' 가 없던 scenegraph에 계산해서 rpe 값 node에 넣어주는 부분
                for i in range(length):   
                    # tmp[0][i].graph['gid'] = i
                    # 서브 그래프를 만든 후에 rpe를 계산? rpe를 계산한 다음에 서브그래프를 만들어야 해당 이미지에서 해당 노드를 더 잘표현하는 것 아닌가?
                    origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args,F0Dict)  # Gs에 Feature 붙임
                    # print(origin_g.nodes(data="rpe"))
                    subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS) # subgraph 로 나눔
                    db.extend(subs)    
                    
                    # node id 갱신 여부 확인 -> 안됨
                    # sys.exit()
                    # print(db[0].nodes(data='name'))
                    # sys.exit()
                    
                    #rpe 재 계산 하는 부분이 있는지 확인
                    print(subs[0].nodes(data=True)) # bbox, generated, tracker, txtemb, name, fid, rpe 
                    print(subs[0].edges(data=True)) # distance, angle_AB, angle_BA, predicate, txtemb, 
                    sys.exit()
                    print(subs[1].nodes(data='name'))
                    sys.eixt()
                    
                    
                    db_idx.extend([str(vId)+ '_' + str(tmp[2][i])+'_'+str(subIdx) for subIdx in range(len(subs))])                
                    print('len(subs): ',len(subs))
                    
                db_reIdx = [i for i in range(len(db))]
                
                sys.exit()
                
        # print("len(db): ", len(db)) 
        # print("len(db): ", len(db_idx)) 
    print("cnt_video: ", cnt_video)
    print("total len(db): ", len(db))
    
    
    
    # query = []
    # # user-defined query images
    # with open("data/seq_g3_4239231056_7645715544.pkl", "rb") as q:
    #     queryDataset = pickle.load(q)
    #     #todo - 여기서 RPE 계산해야함        
        
    #     length = len(queryDataset[0])
    #     if length != 0:
    #         cnt = 0
    #         query_number = [idx for idx in queryDataset[2]]
    #         for i in range(length):   
    #             # queryDataset[i].graph['gid'] = i
    #             origin_g, origin_enc_agg = utils.mkNG2Subs(queryDataset[0][i], args, F0Dict)  # Gs에 Feature 붙임
    #             query.append(origin_g)
    
    query = []
    queryDataset = [[],[],[]]
    for i in range(0, len(db_reIdx), 40):
        queryDataset[0].append(db[i])
        queryDataset[1].append(db_idx[i].split('_')[0])
        queryDataset[2].append(db_idx[i].split('_')[1])
    print(queryDataset)  
       

    length = len(queryDataset[0])
    if length != 0:
        cnt = 0
        query_number = [idx for idx in queryDataset[2]]
        for i in range(length):   
            # queryDataset[i].graph['gid'] = i
            origin_g, origin_enc_agg = utils.mkNG2Subs(queryDataset[0][i], args, F0Dict)  # Gs에 Feature 붙임
            query.append(origin_g)
    
    
    with open("result/query/overlapped.pkl", "wb") as fw:
        pickle.dump([db, db_idx, db_reIdx, query, query_number], fw)
            
                    
    return db, db_idx, db_reIdx, query, query_number


def handcraftQuery():
  
  
  with open("result/1217/vid08_1217_from_db.pkl", "rb") as fr:
    tmp = pickle.load(fr)
  print(len(tmp))
  # 짝수: 그래프 / 홀수: meta
  #[비디오-0:그래프, 1:meta][프레임][서브그래프]
  #동일 비디오, , 다른 서브그래프?
  # print(len(tmp[0])-2)
  db_idx_pair = []
  queryGList = []
  for i in range(len(tmp[0])-2):
    g1 = tmp[0][i].copy()
    g2 = tmp[0][i+1].copy()
    vid_frame_info = (tmp[1][i][0]+" + "+tmp[1][i+1][0])
    print("origin: ", g1 )
    print(g1.nodes(data="name"))
    print("origin: ",g2.nodes(data="name"))

    queryG = g1.copy()
    queryG.remove_node(list(queryG.nodes())[0])
    if len(g2.nodes()) > 2:
      g2.remove_node(list(g2.nodes())[0])
      g2.remove_node(list(g2.nodes())[1])
      print(queryG.nodes(data="name"))
      print(g2.nodes(data="name"))
      
      queryG.add_nodes_from(g2.nodes(data=True))
      queryG.add_edges_from(g2.edges())
      
      #위 방법으로 생성 시 두 그래프 간 edge 로 연결되지 않음. edge의 경우 임의로 추가해줘야함 안할래~
      print("g1: ",g1)
      print(queryG.nodes(data="name"))
      print(g1.nodes(data="name"))
      print(g2.nodes(data="name"))
      print(vid_frame_info)
      queryGList.append(queryG)
      db_idx_pair.append(vid_frame_info)
      
    else:
      print("작아")
      #continue

    # 처음 노드의 경우 겹치는 게 많음
    # g1의 경우 0번 삭제
    # g2의 경우 0번, 2번 삭제하면 여러 개 대량생산생산

  # print(len(queryGList))
  # print((queryGList))
  print("쿼리 그래프 정보")
  print((db_idx_pair))
  
  
  
  # db_idx_pair.append()
  
  
  
  return queryGList, db_idx_pair




def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()
        
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
        data  = pickle.load(f)
        F0Dict = data
    # dataset, db_idx, db_reIdx, querys, query_number = mkOverlappedQuery(args, F0Dict)
    handcraftQuery()
    
    
    
    
    


if __name__ == "__main__":
    torch.set_printoptions(precision=20)
    main()


