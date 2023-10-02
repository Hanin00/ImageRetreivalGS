from utils import utils
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


'''
    0731 - 기존과 달라진 점
    1. 서브그래프를 사용하지 않음
    2. 그래프 형식이 다름 - edge feature 있음;

    db_idx.extend([i]*len(datas))# 기존에는 그래프의 id를 subgraph 개수만큼 생성
    하지만, 지금은 비디오 내 프레임별로 scenegraph 를 생성하므로 파일명과 몇 번째 그래프인지를 표현해주면 됨
'''
def load_dataset_temp(args,F0Dict):
    # with open("data/scenegraph_1/0_6096540713_6096540713.pkl", "rb") as fr:
    #     datas = pickle.load(fr)
    db = []
    db_idx = []
    seeds = 4
    query_number = 5002

    filenames = ["3802296828.json.pkl", "6673828083.json.pkl"]
    for filename in filenames:
        vId = filename.split('.')[0]
        with open("data/scenegraph/"+ filename, "rb") as fr:
            tmp = pickle.load(fr)            
            db_idx.extend([ str(vId)+ '_' + str(i) for i in tmp[2]])
            # print("db_idx: ",db_idx)``
            length = len(tmp[0])        
            # length = 2            
            if length != 0:
                cnt = 0
                for i in range(length):   
                    tmp[0][i].graph['gid'] = i
                    origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)  # Gs에 Feature 붙임
                    db.append(origin_g)
        print("len(db): ", len(db)) 
    print("total len(db): ", len(db))

    query = []
    # user-defined query images
    with open("data/query_graphs.pkl", "rb") as q:
        queryDataset = pickle.load(q)
        #todo - 여기서 RPE 계산해야함
        query_number = 1
        length = len(queryDataset)
        if length != 0:
            cnt = 0
            for i in range(length):   
                queryDataset[i].graph['gid'] = i
                origin_g, origin_enc_agg = utils.mkNG2Subs(queryDataset[i], args, F0Dict)  # Gs에 Feature 붙임
                    
                query.append(origin_g)   

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

def find_duplicate_nodes_and_edges(graph1, graph2):
    common_nodes = set()
    
    # graph1에서 노드 번호와 이름 간의 대응 관계 딕셔너리 생성
    node_number_to_name_graph1 = {node_number: data.get('name') for node_number, data in graph1.nodes(data=True)}
    
    # graph2에서 노드 번호와 이름 간의 대응 관계 딕셔너리 생성
    node_number_to_name_graph2 = {node_number: data.get('name') for node_number, data in graph2.nodes(data=True)}
    
    # 공통 노드 찾기
    for node_number1, data1 in graph1.nodes(data=True):
        for node_number2, data2 in graph2.nodes(data=True):
            if data1.get('name') == data2.get('name'):
                common_nodes.add(data1.get('name'))  # 공통 노드 이름 추가
    
    common_edges = set()
    result = []
    
    for edge1, edge2, data1 in graph1.edges(data=True):
        for edge3, edge4, data2 in graph2.edges(data=True):
            # 엣지의 노드 번호를 노드 이름으로 교체
            node1_name_graph1 = node_number_to_name_graph1.get(edge1)
            node2_name_graph1 = node_number_to_name_graph1.get(edge2)
            node3_name_graph2 = node_number_to_name_graph2.get(edge3)
            node4_name_graph2 = node_number_to_name_graph2.get(edge4)

            if (node1_name_graph1 in common_nodes and node2_name_graph1 in common_nodes) and \
               (node3_name_graph2 in common_nodes and node4_name_graph2 in common_nodes):
            #    print("node 가 동일함")
               if data1['predicate'] == data2['predicate']:
                    # print("predicate도 동일함")
                
                #    data1.get('predicate') == data2.get('predicate'):
                    # 엣지의 'distance' 및 'angleAB' 속성 비교
                    # 
                    if 'distance' in data1 and 'distance' in data2 and \
                    'angle_AB' in data1 and 'angle_AB' in data2:
                        distance_diff = abs(data1['distance'] - data2['distance'])
                        angleAB_diff = abs(data1['angle_AB'] - data2['angle_AB'])
                    else:
                        distance_diff = None
                        angleAB_diff = None
                    
                    if data1['predicate'] == data2['predicate']:
                        predicate = data1['predicate']
                    else:
                        predicate = (data1['predicate'], data2['predicate'])
                    
                    # result.append((predicate, distance_diff, angleAB_diff, (node1_name_graph1, node2_name_graph1, data1), (node3_name_graph2, node4_name_graph2, data2)))
                    result.append((predicate, distance_diff, angleAB_diff, (node1_name_graph1, node2_name_graph1, ), (node3_name_graph2, node4_name_graph2,)))
    return result


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
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
        data  = pickle.load(f)
        F0Dict = data
    dataset, db_idx, querys, query_idx = load_dataset_temp(args, F0Dict)
    
    db_data = utils.batch_nx_graphs_rpe(dataset, None)
    print("db_data: ", db_data)

    # print("db_data: ", len(db_data))

    # model load
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
        
    model = models.GnnEmbedder(args.feature_dim, args.hidden_dim, args)  
    model.to(utils.get_device())
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=utils.get_device()))  
    else:
        return print("model does not exist")

    print("here - feature_extract")
    db_check = [{i[1] for i in d.nodes(data="name")}for d in dataset]
    temp = []
    
    candidate_imgs = []
    model.eval()
    torch.set_printoptions(precision=10)
    with torch.no_grad():
        emb_db_data = model.emb_model(db_data) # [1327,32]
        
        for idx, queryG  in enumerate(querys,9): #i = 쿼리 그래프의 서브 그래프 하나.
            extractTimeStart = time.time()
            query = temp.copy()
            query.append(queryG)
            query = utils.batch_nx_graphs_rpe(query, None)
            query = query.to(utils.get_device())            
            emb_query_data = model.emb_model(query) # 서브그래프 하나에 대한 특징 추출
            
            extractTimeEnd = time.time()
            print("subGraph 하나에 대한 특징 추출 시간 -+ : ", extractTimeEnd - extractTimeStart)
            retreival_start_time = time.time()  # subgraph 하나에 대한 추출 시간
            sim = torch.tensor([torch.sum(emb_query_data * emb_db_data[idx], dim=1).to(utils.get_device()) for idx in range(len(emb_db_data))] ).to(utils.get_device())
            # sim = torch.tensor([torch.dot(emb_query_data, emb_db_data[i]) for i in range(len(emb_db_data))],requires_grad=True).to(utils.get_device())
        
            result_dict = dict(zip(db_idx, sim))
            sorted_items = sorted(result_dict.items(), key=lambda item: item[1])
            
            top_10_items = sorted_items[:10]
            print("10: ")
            print(top_10_items)
            top_10_items = sorted_items[-10:]
            print(":-10 ")
            print(top_10_items)
            
            
            print("Top 10 Sorted db_idx and corresponding results:")
    
            for db_idx, result_value in sorted_items:
                print("db_idx:", db_idx, "Result:", result_value)

            continue
            # graph node 비교 가능하도록 변경 필요
            
            
            q_check = {n[1] for n in queryG.nodes(data="name")} #query graph의 name
            # print("Query num: ",idx)
            print("Q graph nodes :", q_check)
            # print("Q graph: ", queryG)
            print("Q graph edges :", queryG.edges(data=True))
            # print("number of DB subgraph", e.shape)
            # result = [(query_idx+1, i)]
            result = []
            rIdx = 0
            # for n, d in rank[:5]:
            #     print("similarity : {:.5f}".format(d.item()))
            #     result.append((db_idx[n], dataset[n]))
                
            #     #중복된 노드와 엣지 찾기
            #     duplicate_info = find_duplicate_nodes_and_edges(queryG, dataset[n])

            #     # 결과 출력

            #     # print("query idx: ", idx, "  rank: ", rIdx, "db_idx[n]: ", db_idx[n], "전체 db 내 Idx : ", n)
            #     # print("_info: ",duplicate_info)              
       
            #     candidate_imgs.append(db_idx[n])            
            #     rIdx += 1

            # [print("id: ", ranks[0], "\n graphs: ", ranks[1])  for ranks in result]
            showGraph(queryG, 'query', 'query'+str(idx))# query graph 저장
            # [showGraph(rank[1],'ranks', 'qid_'+str(idx)+'-rank_'+ str(rankIdx)+'-id_'+str(rank[0]))  for rankIdx, rank in enumerate (top_10_sorted_items)]           
            #rank[1] : graph / rank[0] : graph db_idx




            retreival_time = time.time() - retreival_start_time
            print("@@@@@@@@@@@@@@@@@retreival_time@@@@@@@@@@@@@@@@@ :", retreival_time)
            
            # # sys.exit()
            # # Check similar/same class count with subgraph in DB
            # checking_in_db = [len(q_check) - len(q_check - i)
            #                   for i in db_check]
            # checking_result = Counter(checking_in_db)
            # print(checking_result)

            # # Check similar/same class with subgraph in DB
            # value_checking_in_db = [
            #     str(q_check - (q_check - i)) for i in db_check]
            # value_checking_result = Counter(value_checking_in_db)
            # print(value_checking_result)
            # print("---^^^---"*3)
            

def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)


if __name__ == "__main__":
    torch.set_printoptions(precision=20)
    main()

