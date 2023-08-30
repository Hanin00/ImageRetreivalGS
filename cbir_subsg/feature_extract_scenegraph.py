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

import networkx as nx
import matplotlib.pyplot as plt





import sys

'''
    Scenegraph에 대한 feature Vector 추출
'''
def load_dataset(args):
    # with open("data/scenegraph_1/0_6096540713_6096540713.pkl", "rb") as fr:
    #     datas = pickle.load(fr)
    db = []
    db_idx = []
    seeds = 4
    query_number = 5002       

    gevpair_dbScenegraphs = os.listdir('data/GEDPair/train/walk4_step3_ged10/') #scenegraph file이 있는 폴더명
    for file_name in gevpair_dbScenegraphs[:2]:
        vID = file_name.split('_')[-2] +'-' + file_name.split('_')[-1]+'-' 
        # vName = file_name.split('_')[1] # video Name
        with open("data/GEDPair/train/walk4_step3_ged10/"+file_name, "rb") as fr:
            tmp = pickle.load(fr)
            # print("*tmp[0]: " , tmp[0][0]) # tmp[0]: g / tmp[1]: g' / tmp[2]: GEV            
            # db.extend(tmp[0]), 
            db.extend(tmp[1])
            db_idx.extend([vID + '_' + str(i) for i in range(len(tmp[0]))])
        # print("db_idx: ", db_idx) # vid_fid -> vid는 Scenegraph 생성시 붙인 번호    

    # user-defined query images
    with open("data/query_ged.pkl", "rb") as q:
        query = pickle.load(q)
        query = query[0][:10] # 비디오별 scenegraph를 포함하고 있기 때문
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
    # print("=*=/="*10)
    # print(dataset[0].nodes(data=True))
    # print("=\=*="*10)
    # sys.exit()
    db_data = utils.batch_nx_graphs_rpe(dataset, None)
    print("db_data: ", db_data)
    # print("db_data: ", len(db_data))

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
        print("==="*20)
        print(len(emb_db_data))
        print(len(emb_db_data[0]))
        sys.exit()

    #     emb_db_data = model.emb_model(db_data)    #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_RBFS.pickle", "wb") as fr:
    # #         pickle.dump(results, fr)
    

def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)


if __name__ == "__main__":
    main()
