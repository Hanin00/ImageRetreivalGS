from utils import utils,  subgraph
from cbir_subsg import models
from cbir_subsg.conf import parse_encoder

from collections import Counter

import os, sys
import torch
import argparse
import pickle
import time

import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np



def main():
    torch.inference_mode()
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    # model load
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
    
    # data load
    max_node = 3
    R_BFS = True

    db = []
    db_idx = []
    
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
        data  = pickle.load(f)
    F0Dict = data

    vIdList = [] # videoId = json file Name
    fIdList = [] # frameId = imageId  
    #subGraphId 가 필요? -> X
    subGFeature = [] # vector # (32, )
            
    with open('data/fileNameList_ordered.pkl', 'rb') as f:
        fileNameList  = pickle.load(f)
    filenames = []
    for item in fileNameList:
        # sliced_item = item[0:8]
        sliced_item = item[8:15] # 23.10.30 추가 요청
        # sliced_item = item[15:]
        filenames.extend(sliced_item)
        
    for filename in filenames:
        vId = filename.split('.')[0]
        print("filename : ",filename)
        with open("data/scenegraph/"+ filename, "rb") as fr:
        # with open("data/scenegraph/"+ filename+'.json.pkl', "rb") as fr:
            tmp = pickle.load(fr)
            length = len(tmp[2])
            
            try:
                # warmup>>> 
                # print("feature_extract - tmp[0][0]: ", tmp[0][0])            
                origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][0], args, F0Dict)  # Gs에 Feature 붙임
                
                # print("feature_extract - origin_g: ", origin_g)
                subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)
                # print("feature_extract - subs: ", subs)
                db_data = utils.batch_nx_graphs_rpe(subs, None)
                emb_db_data = model.emb_model(db_data)       
                # warmup>>> 
            except:
                continue
            
            emb_db_data = model.emb_model(db_data)
            # length = 4 # 쿼리 그래프 시퀀스의 길이

            if length != 0:
                cnt = 0
                timeList = []
                start = time.time()
                print("length : ",length)
                
                for i in range(length):   
                    # print("tmp[0][i]", tmp[0][i])
                    q_start = time.time()
                    tmp[0][i].graph['gid'] = i
                    try: 
                        origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)  # Gs에 Feature 붙임
                        fId = tmp[2][i]        
                        # print(tmp[0][i])
                    except:
                        print(tmp[0][i])
                    
                    # 그래프 하나에 대해 rpe 계산을 해야함
                    # 그 후 rpe 를 거쳐야함
                    # try: 
                    #     subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)
                    #     db_data = utils.batch_nx_graphs_rpe(subs, None)
                    #     db_data = db_data.to(utils.get_device())
                    # except:
                    subs = [origin_g]
                    db_data = utils.batch_nx_graphs_rpe(subs, None)
                    db_data = db_data.to(utils.get_device())
                    
                    with torch.no_grad():
                        emb_db_data = model.emb_model(db_data)
                        # print("쿼리 그래프 하나 당: ", time.time()-q_start)
                        timeList.append(time.time()-q_start)
                        # print("emb_db_data: ", emb_db_data)
                        fIdList.extend([fId] * len(subs))
                        subGFeature.extend(emb_db_data) #feature
                        vIdList.extend([vId] * len(subs))
                # print("평균: ", sum(timeList)/ len(timeList))
                # print("평균(첫번째 제외): ", sum(timeList[1:])/ len(timeList[1:]))
                # # print(timeList)
                # # print("subGFeature: " ,subGFeature)
                # print("쿼리그래프 -> 서브 그래프 -> 추출",time.time() - start)
    # print("len(fIdList) : ",len(fIdList))
    # print("len(subGFeature) : ",len(subGFeature))
    # print("len(vIdList) : ",len(vIdList))
    
    subGFeature_numpy = [item.detach().cpu().numpy() for item in subGFeature]

    df = pd.DataFrame({"vId": vIdList, "fId": fIdList,"subGFeature": subGFeature_numpy,})
    
    df.to_parquet("scenegraphFeature_GAT_edgeattrall_8-15.parquet", engine="pyarrow", compression="gzip")
    # df.to_parquet('subgraphFeature_cbir.parquet', engine='fastparquet', compression='snappy')
    df.to_csv("scenegraphFeature_GAT_edgeattrall_8-15.csv")  
    
    
    
    

def queryFeature():
    
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
        data  = pickle.load(f)
    F0Dict = data
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    # model load
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
    
    max_node = 3
    R_BFS = True

    db = []
    db_idx = []

    vIdList = [] # videoId = json file Name
    fIdList = [] # frameId = imageId  
    #subGraphId 가 필요? -> X
    subGFeature = [] # vector # (32, )
            

    with open("data/query_graphs.pkl", "rb") as fr:
        tmp = pickle.load(fr)
        length = len(tmp[0])
        
        # fId = tmp[2][i] 
        '''
        - 그래프의 크기 - 노드 3개 -> 서브 그래프 생성 필요 X -> 더미 데이터 생성????
        쿼리 그래프의 시퀀스 길이: 4개 
        특징 추출 시간 비교
        -> 지금처럼 나오는지 확인; 시퀀스에서 첫번째꺼가 자꾸 오래 걸리는지 원인 파악
        모델load -> warm up 
        '''
        
        if length != 0:
            cnt = 0            
            start = time.time()
            for i in range(length):
                print(tmp[i])
                q_start = time.time()
                origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[i], args, F0Dict)  # Gs에 Feature 붙임 
                subs = subgraph.make_subgraph(origin_g, max_node, False, R_BFS)
                db_data = utils.batch_nx_graphs_rpe(subs, None)
                db_data = db_data.to(utils.get_device())
                with torch.no_grad():
                    e_start = time.time()
                    emb_db_data = model.emb_model(db_data)
                    print("emb 만: ", time.time() - e_start)
                    # print("emb_db_data: ", emb_db_data)
                    fIdList.extend([i] * len(subs))
                    subGFeature.extend(emb_db_data) #feature
                    vIdList.extend([0] * len(subs))
                
                print(len(db_data))
                sys.exit()
                    
                print("쿼리 그래프 하나 당: ", time.time()-q_start)
            print("쿼리그래프 -> 서브 그래프 -> 추출",time.time() - start)
            
    # print("len(fIdList) : ",len(fIdList))
    # print("len(subGFeature) : ",len(subGFeature))
    # print("len(vIdList) : ",len(vIdList))
    # print("subGFeature : ", subGFeature)
    # print("fIdList : ",fIdList)
    
    subGFeature_numpy = [item.detach().cpu().numpy() for item in subGFeature]
    
    df = pd.DataFrame({"vId": vIdList, "fId": fIdList,"subGFeature": subGFeature_numpy,})
    
    # df.to_parquet("querygraphFeature_alledgeattr_subg.parquet", engine="pyarrow", compression="gzip")
    # # df.to_parquet('subgraphFeature_cbir.parquet', engine='fastparquet', compression='snappy')
    # df.to_csv("querygraphFeature_alledgeattr_subg.csv")  


    
if __name__ == "__main__":
    # queryFeature()
    main()