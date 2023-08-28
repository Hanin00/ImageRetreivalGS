from utils import utils
from model import models
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

'''
    0731 - 기존과 달라진 점
    1. 서브그래프를 사용하지 않음
    2. 그래프 형식이 다름 - edge feature 있음;

    db_idx.extend([i]*len(datas))# 기존에는 그래프의 id를 subgraph 개수만큼 생성
    하지만, 지금은 비디오 내 프레임별로 scenegraph 를 생성하므로 파일명과 몇 번째 그래프인지를 표현해주면 됨
'''
def load_dataset(args):
    with open("data/GEDPair/train/walk4_step3_ged5/walk4_step3_ged5_1604_50.pkl", "rb") as fr:
        datas = pickle.load(fr)
    db = []
    db_idx = []

    seeds = 4
    query_number = 5002                         #todo meta data 기준으로 걸러야함
    for i in range(len(datas)):
        if query_number == i:
            continue
        db.extend(datas[0])
        #db_idx.extend([i]*len(datas))# 기존에는 그래프의 id를 subgraph 개수만큼 생성했

    # user-defined query images
    with open("data/query_ged.pkl", "rb") as q:
        query = pickle.load(q)
        query = query[0][:20] # 비디오별 scenegraph를 포함하고 있기 때문
        # print("query: ",query)
        # query, queryFeatList = mkG.mkSubs(querys[0], args, seeds)
        query_number = 1
    return db, db_idx, query, query_number


def feature_extract(args):
    ''' Extract feature from subgraphs
    It extracts all subgraphs feature using a trained model.
    and then, it compares DB subgraphs and query subgraphs and finds0
    5 candidate DB subgraphs with similar query subgraphs.
    Finally, it counts all candidate DB subgraphs and finds The highest counted image.
    '''
    # max_node = 3
    # R_BFS = True
    # ver = 2
    #dataset, db_idx, querys, query_idx = load_dataset(max_node, R_BFS)
    dataset, db_idx, querys, query_idx = load_dataset(args)
    db_data = utils.batch_nx_graphs_rpe(dataset, None)

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
        # emb_db_data = model.emb_model(db_data)

        emb_db_data = model.emb_model(db_data)
        for i in querys: #i = 쿼리 그래프의 서브 그래프 하나.
            query = temp.copy()
            query.append(i)
            query = utils.batch_nx_graphs_rpe(query, None)
            query = query.to(utils.get_device())

            extractTimeStart = time.time()
            emb_query_data = model.emb_model(query) # 서브그래프 하나에 대한 특징 추출
            extractTimeEnd = time.time()
            print("subGraph 하나에 대한 특징 추출 시간 -+ : ", extractTimeEnd - extractTimeStart)
            print(emb_db_data.data)
            print(emb_db_data.data.size())
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
                print("db_idx: ", db_idx) #<- 이거 메타 데이터를 어떻게 해야하지..?
                # sys.exit()
                # print("DB img :", db_idx.item())
                # print("DB img id :", db_idx.item())
                print("similarity : {:.5f}".format(d.item()))
                print("DB graph nodes :", [n])
                sys.exit()
                # print("DB img :", db_idx.item())
                # print("rank graph id - n:", n)
                # print("rank graph id - d:", d)
                # sys.exit()
                result.append((db_idx[n]+1, dataset[n]))

                candidate_imgs.append(db_idx[n]+1)

            results.append(result)
            retreival_time = time.time() - retreival_start_time
            print("@@@@@@@@@@@@@@@@@retreival_time@@@@@@@@@@@@@@@@@ :", retreival_time)
            
            # sys.exit()
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


def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)


if __name__ == "__main__":
    main()
