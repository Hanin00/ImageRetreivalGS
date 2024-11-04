<<<<<<< HEAD
from utils import utils,  subgraph
=======
from utils import utils
>>>>>>> master
from cbir_subsg import models
from cbir_subsg.conf import parse_encoder

from collections import Counter

<<<<<<< HEAD
import os, sys
=======
import os
>>>>>>> master
import torch
import argparse
import pickle
import time
<<<<<<< HEAD
=======
import tqdm

import numpy as np
>>>>>>> master

import networkx as nx
import matplotlib.pyplot as plt

<<<<<<< HEAD
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
=======
import time
import sys

'''
    Scenegraph에 대한 feature Vector 추출
    1. SceneGraph load
    2. SceneGraph 에 대해 RPE 값 계산 - 근데 이미 GED에서 한 거 있어서 안해도 되지 않나??
        -> 파일이 너무 많아서 cpu core 개수로 묶었는데, 한 묶음당 파일 크기의 총량이 비슷한 크기가 되도록 정렬했음
            이에대해 파일명을 20개 단위로 나눴었음. 
            GEDPair/train/walk4_step3_ged10/ 아래에는, sorting 된 것들에서 각 파일명 묶음별로 20개까지 슬라이싱해서 만든 것
            -> 이 파일들의 개수가 1,000,000개가 되는지 확인할 것 @@@
    3. 해당 파일들을 불러서 feature 추출 시 pkl 보다 더 압축효율..?이 좋은 파일을 사용하는 게 좋을 것 같다. parque 같은거!
        시간 확인
'''



import sys, os
import pickle
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import networkx as nx
import multiprocessing as mp

import random
import math
import os, sys

import pickle
import multiprocessing



# from utils.mkGraphRPE import *
# from surel_gacc import run_walk

# def mkMergeGraph(S, K, gT, nodeNameDict, F0dict, nodeIDDict):
#     merged_K = np.concatenate([np.asarray(k) for k in K]).tolist()
#     # print("merged_K: ",merged_K)
#     merged_K = [nodeIDDict[i] for i in merged_K]
#     # print("after merged_K: ", merged_K)
    
#     sum_dict = {}
#     count_dict = {}
#     for k, gf in zip(merged_K, gT):
#         if k in sum_dict:
#             sum_dict[k] += gf
#             count_dict[k] += 1
#         else:
#             sum_dict[k] = gf
#             count_dict[k] = 1 
#     gT_mean = {k: sum_dict[k] / count_dict[k] for k in sum_dict}
    
#     return gT_mean 


# def mkNG2Subs(G, args, F0dict):
#     # print("---- mkNG2Subs ----")
#     nmDict = dict((int(x), y['name'] ) for x, y in G.nodes(data=True)) 
#     Gnode = list(G.nodes())  
#     G_full = csr_matrix(nx.to_scipy_sparse_array(G))

#     ptr = G_full.indptr
#     neighs = G_full.indices
#     num_pos, num_seed, num_cand = len(set(neighs)), 100, 5
#     candidates = G_full.getnnz(axis=1).argsort()[-num_seed:][::-1]
#     # print("candidates : ", candidates)
#     rw_dict = {}
#     B_queues  = []

#     batchIdx, patience = 0, 0
#     pools = np.copy(candidates)

#     np.random.shuffle(B_queues)
#     B_queues.append(sorted(run_sample(ptr,  neighs, pools, thld=1500)))
#     B_pos = B_queues[batchIdx]

#     B_w = [b for b in B_pos if b not in rw_dict]
#     if len(B_w) > 0:
#         walk_set, freqs = run_walk(ptr, neighs, B_w, num_walks=args.num_walks, num_steps=args.num_steps - 1, replacement=True)
#     node_id, node_freq = freqs[:, 0], freqs[:, 1]
#     rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))
#     batchIdx += 1

#     S, K, F = zip(*itemgetter(*B_pos)(rw_dict))

#     F = np.concatenate(F)
#     mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F])) 
#     gT = mkGutils.normalization(mF, args)

#     listA = [a.flatten().tolist() for a in K] 
#     flatten_listA = list(itertools.chain(*listA)) 

#     gT_concatenated = torch.cat((gT, gT), axis=1)
#     enc_agg = torch.mean(gT_concatenated, dim=0)

#     nodeIDDict = dict(zip(candidates, Gnode))
#     rpeDict = mkMergeGraph (S, K, gT, nmDict, F0dict, nodeIDDict)
#     for nodeId in list(G.nodes()):
#         G.nodes()[nodeId]['rpe']  = rpeDict[nodeId]

#     return G, enc_agg
 

# #-----^^^^ RPE 계산 ^^^^-------------------

#model load, embedding Vector return
def feature_extract(args, graphs):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    # model = models.GnnEmbedder(1, args.hidden_dim, args)  
    model = models.GnnEmbedder(args.feature_dim, args.hidden_dim, args)  
    model.to(utils.get_device())
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))  
    else:
        return print("model does not exist")
    model.eval()
    print(model)
    sys.exit()
    
    with torch.no_grad():
        emb_db_data = model.emb_model(graphs)
        return emb_db_data



def main(args): 
    
    # model load <-병렬 처리 시 고려
    # ------------ rpe 계산 ---------------------
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
        data  = pickle.load(f)
        F0Dict = data
    folder_path = "data/scenegraph"
    filenames = os.listdir(folder_path)
    rpeGraphs = []
    filenames = filenames[:2]
    for filename in filenames:
        print("filename: ", filename)
        fpath = "data/scenegraph/"+str(filename)    
        with open(fpath, 'rb') as file:
            data = pickle.load(file)
        dataset = data[0] #video 내 graphs
        length = len(dataset)
        
        if length != 0:
            cnt = 0
            for i in range(length):   
                dataset[i].graph['gid'] = 0
                origin_g, origin_enc_agg = utils.mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 Feature 붙임 #txtemb 는 10개씩 있음
                rpeGraphs.append(origin_g)
                
    try:
        rpeGraphs = utils.batch_nx_graphs_rpe(rpeGraphs, None)
    except:
        print("rpeGraphs: ",rpeGraphs[-1])


    inference_start = time.time()
    emb_db_data = feature_extract(args, rpeGraphs) # 1293개 그래프에 대해 0.9680685997009277
    # print("emb_db_data: ",emb_db_data)
    # sys.exit()
    inference_end = time.time()
    print("inference time: ", inference_end-inference_start)
    print("len(emb_db_data): " ,len(emb_db_data))
    print("len(emb_db_data[0]): " ,len(emb_db_data[0]))
    print("(emb_db_data[0]): " ,(emb_db_data[0]))
    
    
    

# ------------ rpe 계산 ---------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    margs = parser.parse_args()
    main(margs)

>>>>>>> master
