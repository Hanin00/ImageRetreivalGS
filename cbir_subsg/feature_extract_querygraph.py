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


'''
    노드 3개, edge 3개로 이뤄진 쿼리 그래프의 특징 추출

'''
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
            
    with open("data/seq_g3_4239231056_7645715544.pkl", "rb") as fr:
        tmp = pickle.load(fr)        
        length = len(tmp[0])
        dbdata = []
        if length != 0:
            cnt = 0            
            start = time.time()
            for i in range(length):
                q_start = time.time()
                origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)  # Gs에 Feature 붙임 
                dbdata.append(origin_g)
                fIdList.append(tmp[2][i])
                vIdList.append(tmp[1][i])
                

        db_data = utils.batch_nx_graphs_rpe(dbdata, None)
        db_data = db_data.to(utils.get_device())
        with torch.no_grad():
            e_start = time.time()
            emb_db_data = model.emb_model(db_data)
            print("emb 만: ", time.time() - e_start)
            # print("emb_db_data: ", emb_db_data)
            
            subGFeature.extend(emb_db_data) #feature
                                
                
    subGFeature_numpy = [item.detach().cpu().numpy() for item in subGFeature]
    
    df = pd.DataFrame({"vId": vIdList, "fId": fIdList,"subGFeature": subGFeature_numpy,})
    
    df.to_parquet("querygraphFeature_alledgeattr_node3_4239-7645.parquet", engine="pyarrow", compression="gzip")
    # df.to_parquet('subgraphFeature_cbir.parquet', engine='fastparquet', compression='snappy')
    df.to_csv("querygraphFeature_alledgeattr_node3_4239-7645.csv")  


    
if __name__ == "__main__":
    queryFeature()
    # main()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