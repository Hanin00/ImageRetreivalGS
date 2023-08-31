from utils import utils
from model import models
from config.config import parse_encoder

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

'''
    0731 - 기존과 달라진 점
    1. 서브그래프를 사용하지 않음
    2. 그래프 형식이 다름 - edge feature 있음;

    db_idx.extend([i]*len(datas))# 기존에는 그래프의 id를 subgraph 개수만큼 생성
    하지만, 지금은 비디오 내 프레임별로 scenegraph 를 생성하므로 파일명과 몇 번째 그래프인지를 표현해주면 됨
'''
def load_dataset(args):
    # with open("data/scenegraph_1/0_6096540713_6096540713.pkl", "rb") as fr:
    #     datas = pickle.load(fr)
    db = []
    db_idx = []
    seeds = 4
    query_number = 5002       
    
    # ------- scenegraph ----------------
    # dbScenegraphs = os.listdir('data/scenegraph_1/') #scenegraph file이 있는 폴더명
    # for file_name in dbScenegraphs[:20]:
    #     vID = file_name.split('_')[0]
    #     # vName = file_name.split('_')[1] # video Name
    #     with open("data/scenegraph_1/"+file_name, "rb") as fr:
    #         tmp = pickle.load(fr)
    #         db.extend(*tmp[0])
    #         db_idx.extend([vID + '_' + str(i) for i in range(len(*tmp[0]))])
    #         # print("db_idx: ", db_idx) # vid_fid -> vid는 Scenegraph 생성시 붙인 번호
    # ------- scenegraph ---------------- -> rpe가 없어서 사용 불가. 계산하는 부분 추가 해야 함


    gevpair_dbScenegraphs = os.listdir('data/GEDPair/train/walk4_step3_ged10/') #scenegraph file이 있는 폴더명   
    for file_name in gevpair_dbScenegraphs:
        vID = file_name.split('_')[-2] +'-' + file_name.split('_')[-1]+'-' 
        # vName = file_name.split('_')[1] # video Name
        with open("data/GEDPair/train/walk4_step3_ged10/"+file_name, "rb") as fr:
            tmp = pickle.load(fr)
            db.extend(tmp[0][0])
            db_idx.extend([vID + '_' + str(i) for i in range(len(tmp[0]))])
    
    # user-defined query images
    with open("data/query_graphs.pkl", "rb") as q:
        query = pickle.load(q)
        query_number = 1
    return db, db_idx, query, query_number


def load_dataset_temp(args):
    # with open("data/scenegraph_1/0_6096540713_6096540713.pkl", "rb") as fr:
    #     datas = pickle.load(fr)
    db = []
    db_idx = []
    seeds = 4
    query_number = 5002
    with open("data/1graph_per_video.pkl", "rb") as fr:
        tmp = pickle.load(fr)
        db.extend(tmp) #scenegraph 에 rpe 계산한 것
        db_idx.extend([i for i in range(len(db))])
    # print("db_idx: ", db_idx) # vid_fid -> vid는 Scenegraph 생성시 붙인 번호
    print("len(datset) : ", len(db))

    # user-defined query images
    with open("data/query_graphs.pkl", "rb") as q:
        query = pickle.load(q)
        query_number = 1
    return db, db_idx, query, query_number



def showGraph(graph, type, title):
    #query graph 시각화, 저장
    plt.figure(figsize = (8, 8))
    pos = nx.spring_layout(graph, k = 0.8)
    node_labels = nx.get_node_attributes(graph, 'name')
    edge_labels = nx.get_edge_attributes(graph, 'predicate')
    # print("edge_labels: ", edge_labels)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels = edge_labels)
    nx.draw(graph, pos, node_size = 400, node_color = 'blue',)
    plt.show()
    plt.title(type+'-'+title)
    plt.savefig('result/'+type+'/'+title+'.png',
    facecolor='#eeeeee',
    edgecolor='black',
    format='png', dpi=200)


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
    # dataset, db_idx, querys, query_idx = load_dataset(args)
    dataset, db_idx, querys, query_idx = load_dataset_temp(args)
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
        for idx, queryG  in enumerate(querys): #i = 쿼리 그래프의 서브 그래프 하나.
            print("---vvv---"*3)

            query = temp.copy()
            query.append(queryG)
            query = utils.batch_nx_graphs_rpe(query, None)
            query = query.to(utils.get_device())

            extractTimeStart = time.time()
            emb_query_data = model.emb_model(query) # 서브그래프 하나에 대한 특징 추출
            extractTimeEnd = time.time()
            
            print("subGraph 하나에 대한 특징 추출 시간 -+ : ", extractTimeEnd - extractTimeStart)

            retreival_start_time = time.time()  # subgraph 하나에 대한 추출 시간
            e = torch.sum(torch.abs(emb_query_data - emb_db_data), dim=1)
            rank = [(i, d) for i, d in enumerate(e)]
            rank.sort(key=lambda x: x[1])
            q_check = {n[1] for n in queryG.nodes(data="name")} #query graph의 name
            print("Query num: ",idx)
            print("Q graph nodes :", q_check)
            print("Q graph: ", queryG)
            print("Q graph edges :", queryG.edges(data=True))
            print("number of DB subgraph", e.shape)
            # result = [(query_idx+1, i)]
            result = []
            for n, d in rank[:5]:
                print("similarity : {:.5f}".format(d.item()))
                print("n : ",n)
                result.append((db_idx[n], dataset[n]))
                print("db_idx[n] = filename: ", db_idx[n]) 
                print("dataset[n] = graph: ", dataset[n]) #dataset[n] = graph
                print("result graph edges: ", dataset[n].edges(data=True))
                
                candidate_imgs.append(db_idx[n])            

            # [print("id: ", ranks[0], "\n graphs: ", ranks[1])  for ranks in result]
            showGraph(queryG, 'query', 'qurey_'+str(idx))# query graph 저장
            [showGraph(rank[1],'ranks', 'qid_'+str(idx)+'-rank_'+ str(i)+'-id_'+str(rank[0]))  for i, rank in enumerate (result)]
           

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
            print("---^^^---"*3)
            
    # Final image rank -> 서브 그래프에 대해서 이미지 검색을 한 것이 아니라서 해당 Counter는 필요 없음
    # imgs = Counter(candidate_imgs)
    # print(imgs)

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

