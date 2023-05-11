from utils import utils
from model import models
#from common import subgraph
# from utils import mkGraphRPE as mkG
from utils import mkGraphRPE_0511 as mkG
from utils import gev_generator # gpu 06

from config.config import parse_encoder

from collections import Counter

import os
import torch
import argparse
import pickle
import time
import tqdm

import numpy as np


import sys


def feature_extract(args):
    ''' Extract feature from subgraphs
    It extracts all subgraphs feature using a trained model.
    and then, it compares DB subgraphs and query subgraphs and finds0
    5 candidate DB subgraphs with similar query subgraphs.
    Finally, it counts all candidate DB subgraphs and finds The highest counted image.
    '''
    # max_node = 3
    R_BFS = True
    ver = 2
    #dataset, db_idx, querys, query_idx = load_dataset(max_node, R_BFS)
    dataset, db_idx, querys, query_idx = load_dataset(args)


    db_data = utils.batch_nx_graphs_rpe(dataset, None)
    db_data = db_data.to(utils.get_device())

    # model load
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    # model = models.GnnEmbedder(1, args.hidden_dim, args)  
    model = models.GnnEmbedder(args.feature_dim, args.hidden_dim, args)  
    model.to(utils.get_device())
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))
              
    else:
        return print("model does not exist")

    db_check = [{i[1] for i in d.nodes(data="name")}for d in dataset]
    temp = []
    results = []
    candidate_imgs = []
    model.eval()
    with torch.no_grad():
        emb_db_data = model.emb_model(db_data)
        
        for i in querys: #i = 쿼리 그래프의 서브 그래프 하나. 
            query = temp.copy()
            query.append(i)
            query = utils.batch_nx_graphs(query, None)
            query = query.to(utils.get_device())

            extractTimeStart = time.time()
            emb_query_data = model.emb_model(query) # 서브그래프 하나에 대한 특징 추출
            extractTimeEnd = time.time()
            print("subGraph 하나에 대한 특징 추출 시간  : ", extractTimeEnd - extractTimeStart)
            print(emb_db_data.data)
            sys.exit()

            print(emb_db_data.shape)
            retreival_start_time = time.time()  # subgraph 하나에 대한 추출 시간
            e = torch.sum(torch.abs(emb_query_data - emb_db_data), dim=1)
            rank = [(i, d) for i, d in enumerate(e)]
            rank.sort(key=lambda x: x[1])
            q_check = {n[1] for n in i.nodes(data="name")}
            print("Q graph nodes :", q_check)
            print("number of DB subgraph", e.shape)
            # result = [(query_idx+1, i)]
            result = []
            for n, d in rank[:5]:
                print("DB img id :", db_idx[n]+1)
                print("similarity : {:.5f}".format(d.item()))
                print("DB graph nodes :", db_check[n])
                result.append((db_idx[n]+1, dataset[n]))

                candidate_imgs.append(db_idx[n]+1)

            results.append(result)
            retreival_time = time.time() - retreival_start_time
            print("@@@@@@@@@@@@@@@@@retreival_time@@@@@@@@@@@@@@@@@ :", retreival_time)

            # Check similar/same class count with subgraph in DB
            checking_in_db = [len(q_check) - len(q_check - i)
                              for i in db_check]
            checking_result = Counter(checking_in_db)
            print(checking_result)

            # Check similar/same class with subgraph in DB
            value_checking_in_db = [
                str(q_check - (q_check - i)) for i in db_check]
            value_checking_result = Counter(value_checking_in_db)
            print(value_checking_result)
            print("==="*20)
    # Final image rank
    imgs = Counter(candidate_imgs)
    print(imgs)

    # Store result
    # if R_BFS:
    #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_RBFS.pickle", "wb") as fr:
    #         pickle.dump(results, fr)
    # else:
    #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_dense.pickle", "wb") as fr:
    #         pickle.dump(results, fr)



#def load_dataset(max_node, R_BFS):
def load_dataset(args):
    ''' Load subgraphs
    Load Scene Graph and then, Creat subgraphs from Scene Graphs.
    First, it reads scene graphs of Visual Genome and then, it makes subgraphs
    Second, it selects query image and then, it makes subgraphs
    ps) It can use user-defined query images

    max_node: When subgraphs create, It configures subgraph size.
    R_BFS: When subgraphs create, Whether it`s R_BFS mothod or not.

    Return
    db: Subgraphs in database
    db_idx: Index image of subgraphs
    query: Query subgraphs/subgraph
    query_number: Query subgraph number
    '''
    with open('dataset/totalEmbDictV3_x100.pickle', 'rb') as f:  
       f0Dict  = pickle.load(f)
    with open("dataset/v3_x1000.pickle", "rb") as fr:
        datas = pickle.load(fr)
    datas = datas[:100]

    db = []
    db_idx = []

# DB내 동일한 이미지의 번호 제외
#    # Make subgraph from scene graph of Visual Genome
#    query_number = 5002 
#    for i in range(len(datas)):
#        if query_number == i:
#            continue
#        subs = mkGraphRPE.mkSubs(datas[i], max_node, False, R_BFS)
# 
#        db.extend(subs)
#        db_idx.extend([i]*len(subs))

    #seeds = np.random.choice(pools, 5, replace=False)
    seeds = 4
    query_number = 5002                         #todo meta data 기준으로 걸러야함
    pools = 5
    seeds = np.random.choice(pools, 5, replace=False)
    # for i in range(len(datas)):
    for originGId, G in tqdm.tqdm(enumerate(datas)):
        if query_number == originGId:
            continue
       # subGList, subGFeatList = mkGraphRPE.mkSubs(datas[i], num_walks, num_steps, seeds)
        if(len(G.nodes()) != 0):
            subGList, subGFeatList = mkG.mkSubs(G, args, seeds)
            # metaData.append((originGId, len(subGList)))
        db.extend(subGList)
        db_idx.extend([originGId]*len(subGList))

    # query = mkGraphRPE.mkSubs(datas[query_number], max_node, False, True)

    # Select query image
    # user-defined query images
    with open("dataset/query_road_0819.pickle", "rb") as q:
        querys = pickle.load(q)
    query, queryFeatList = mkG.mkSubs(querys[0], args, seeds)
    query_number = 1
    return db, db_idx, query, query_number


def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)


if __name__ == "__main__":
    main()
