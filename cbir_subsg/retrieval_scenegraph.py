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
                # 'rpe' 가 없던 scenegraph에 계산해서 rpe 값 node에 넣어주는 부분
                for i in range(length):   
                    tmp[0][i].graph['gid'] = i
                    origin_g, origin_enc_agg = utils.mkNG2Subs(tmp[0][i], args, F0Dict)  # Gs에 Feature 붙임
                    db.append(origin_g)
        print("len(db): ", len(db)) 
    print("total len(db): ", len(db))
    print()


    #QUERY ----------------------
    # query = []
    # user-defined query images
    # with open("data/query_graphs_10.pkl", "rb") as q:
    #     queryDataset = pickle.load(q)
    #     #todo - 여기서 RPE 계산해야함
    #     query_number = 1
    #     length = len(queryDataset)
    #     length = 2
        
    #     if length != 0:
    #         cnt = 0
    #         for i in range(length):   
    #             queryDataset[i].graph['gid'] = i
    #             origin_g, origin_enc_agg = utils.mkNG2Subs(queryDataset[i], args, F0Dict)  # Gs에 Feature 붙임
    #             query.append(origin_g)   
    #---------------------- QUERY 
    
    #  DB 내에 있는 서브 그래프를 쿼리로 했을 때, 동일하게, 제일 가깝게 추출하는지 확인    
    query = []
    query.append(db[3])
    print("query Idx: ", db_idx[3])
    query.append(db[7])
    print("query Idx: ", db_idx[7])
    query.append(db[-7])
    print("query Idx: ", db_idx[-7])
    query.append(db[-3])
    print("query Idx: ", db_idx[-3])
    
    
    print("total len(query): ", len(query))
    # sys.exit()
    
    
    # with open("data/query_graphs_10.pkl", "rb") as q:
    #         queryDataset = pickle.load(q)
    #         #todo - 여기서 RPE 계산해야함
    #         query_number = 1
    #         length = len(queryDataset)
    #         length = 2
            
    #         if length != 0:
    #             cnt = 0
    #             for i in range(length):   
    #                 queryDataset[i].graph['gid'] = i
    #                 origin_g, origin_enc_agg = utils.mkNG2Subs(queryDataset[i], args, F0Dict)  # Gs에 Feature 붙임
    #                 query.append(origin_g)   


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
    torch.set_printoptions(precision=15)
    with torch.no_grad():
        queryg = querys[1]
        # print("queryg: ",queryg)
        # # sys.exit()
        queryG = [queryg]
        query = utils.batch_nx_graphs_rpe(queryG, None) ### <- 여기서 문제 생긴 것 같음
        emb_query_data1 = model.emb_model(query)
        
        print("query: ", query)
        print("emb_query_data - 1 : ", emb_query_data1.to(utils.get_device()) )

        emb_query_data1_2 = model.emb_model(query)
        cos = nn.CosineSimilarity(dim = 1)
        print("query: ", query)
        print("cos: ", cos(emb_query_data1,emb_query_data1_2))
        print("emb_query_data - 2 : ", emb_query_data1_2.to(utils.get_device()) )

        sim = torch.sum(emb_query_data1 * emb_query_data1, dim=1).to(utils.get_device()) 
        # sim = torch.sum(emb_query_data1 * emb_query_data1, dim=1).to(utils.get_device()) 
        # sim = torch.tensor([torch.sum(emb_query_data * emb_db_data[idx], dim=1).to(utils.get_device()) for idx in range(len(emb_db_data))] ).to(utils.get_device())
        print("emb_query_data1 <-> 1-2 sim : ",sim)
        


        queryG = [queryg]

        # query = utils.batch_nx_graphs_rpe(queryG, None) ### <- 여기서 문제 생긴 것 같음
        # query = query.to(utils.get_device())    
        emb_query_data2 = model.emb_model(query)
            
        print("emb_query_data - 2 : ",emb_query_data2.to(utils.get_device()) )
        # 흠.. 다른 emb 값이 나옴. 왜지!!! Batch에서 Graph 를 
        sim = torch.sum(emb_query_data1 * emb_query_data2, dim=1).to(utils.get_device()) 
        # sim = torch.tensor([torch.sum(emb_query_data * emb_db_data[idx], dim=1).to(utils.get_device()) for idx in range(len(emb_db_data))] ).to(utils.get_device())
        print("sim : ",sim)
        cos = nn.CosineSimilarity(dim = 1)
        print("cos: ", cos(emb_query_data1,emb_query_data2))
        
        another_qg = dataset[-1]
        aqG = [another_qg]
        aqG_batch = utils.batch_nx_graphs_rpe(aqG, None) ### <- 여기서 문제 생긴 것 같음
        emb_query_data3 = model.emb_model(aqG_batch)
        
        print("query: ", query)
        print("emb_query_data3 - 1 : ", emb_query_data3.to(utils.get_device()) )
        
        sim2 = torch.sum(emb_query_data1 * emb_query_data3, dim=1).to(utils.get_device()) 
        # sim = torch.tensor([torch.sum(emb_query_data * emb_db_data[idx], dim=1).to(utils.get_device()) for idx in range(len(emb_db_data))] ).to(utils.get_device())
        print("sim2 : ",sim2)
        cos = nn.CosineSimilarity(dim = 1)
        print("cos: ", cos(emb_query_data1,emb_query_data3))
        
        
        sys.exit()
        
        
        
        
        
        emb_db_data = model.emb_model(db_data) # [1327,32]
        for idx, queryG  in enumerate(querys): #i = 쿼리 그래프의 서브 그래프 하나.
            extractTimeStart = time.time()
            query = temp.copy()
            query.append(queryG)
            query = utils.batch_nx_graphs_rpe(query, None) ### <- 여기서 문제 생긴 것 강늠
            query = query.to(utils.get_device())            
            emb_query_data = model.emb_model(query) # 서브그래프 하나에 대한 특징 추출
            
            extractTimeEnd = time.time()
            print("subGraph 하나에 대한 특징 추출 시간 -+ : ", extractTimeEnd - extractTimeStart)
            retreival_start_time = time.time()  # subgraph 하나에 대한 추출 시간
            #similarity
            sim = torch.tensor([torch.sum(emb_query_data * emb_db_data[idx], dim=1).to(utils.get_device()) for idx in range(len(emb_db_data))] ).to(utils.get_device())
            result_dict = dict(zip(db_idx, sim.cpu().numpy()))  # 토치 텐서를 넘파이 배열로 변환
            sorted_items = sorted(result_dict.items(), key=lambda item: item[1])  # 값을 기준으로 오름차순 정렬
            # print(sorted())
            # print(list(sorted(result_dict, key=lambda x: result_dict[x]))[:10])
            
            print("Top 10 Sorted db_idx and corresponding results:")
            print(sorted_items[:10])
            
            # for db_idx_sorted, result_value in sorted_items:
            #      print("db_idx:", db_idx_sorted, "Result:", result_value)
            # continue
            # # graph node 비교 가능하도록 변경 필요
            
            q_check = {n[1] for n in queryG.nodes(data="name")} #query graph의 name
            # print("Query num: ",idx)
            print("Q graph nodes :", q_check)
            # print("Q graph: ", queryG)
            print("Q graph edges :", queryG.edges(data=True))
            # print("number of DB subgraph", e.shape)
            # result = [(query_idx+1, i)]
            result = []
            rIdx = 0
            
            duplicate_info_info = [] # vId_fId, similarity, graph
            duplicate_info_result = [] # edge distance, node name?
            
            
            # 상위 10개만 확인
            for n, d in sorted_items[:10]:
                print("n: ", n)  #vId_fId
                print("sim: ", d)  #vId_fId
                print("db_idx.index(n) : ",db_idx.index(n)) #db 내 idx 
                print("dataset[db_idx.index(n)] : ",dataset[db_idx.index(n)]) #graph

                
            continue    
            
            

            # 전체 결과에서 중복된 노드가 있는지 확인하고, 얼마나 차이나는지 확인            
            for n, d in sorted_items:
                # print("n: ", n)  #vId_fId
                # print("db_idx.index(n) : ",db_idx.index(n)) #db 내 idx 
                # print("dataset[db_idx.index(n)] : ",dataset[db_idx.index(n)]) #graph
                # print("d: ", d) # similarity
                # print("d: ", str(d)[:7]) # similarity
                # sys.exit()
                
                # print(db_idx[db_idx.index(n)])
                # # print(dataset[db_idx.index(n)])
                
                # print("similarity : {:.5f}".format(d.item()))
                # print("db_idx[n]: ", db_idx.index(n))
                # print("dataset[db_idx.index(n)]: ", dataset[db_idx.index(n)])
                result.append([n ,  dataset[db_idx.index(n)]]) # 
                
                #중복된 노드와 엣지 찾기
                duplicate_info = find_duplicate_nodes_and_edges(queryG, dataset[db_idx.index(n)])
                
                if(len(duplicate_info)) != 0:
                    duplicate_info_info.append((n, d, dataset[db_idx.index(n)]))
                    duplicate_info_result.append((n, duplicate_info))
                    
                # 결과 출력
                # print("query idx: ", idx, "  rank: ", rIdx, "db_idx[n]: ", n , "전체 db 내 Idx : ", db_idx.index(n))
                # print("_info: ",duplicate_info)              
       
                # candidate_imgs.append(db_idx[n])            
                candidate_imgs.append(n) # vId_fId
                rIdx += 1

            # [print("id: ", ranks[0], "\n graphs: ", ranks[1])  for ranks in result]
            showGraph(queryG, 'query', 'query'+str(idx))# query graph 저장
            # print("len(duplicate_info_info) : " ,len(duplicate_info_info))
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # sys.exit()
            if len(duplicate_info_info)!=0:
                if len(duplicate_info_info) >= 5:
                    duplicate_info_info = duplicate_info_info[:5]
                    duplicate_info_result = duplicate_info_result[:5]
                    
                [showGraph(rank[2],'ranks', 'qid_'+str(idx)+'-rank_'+ str(rankIdx)+'-id_'+str(rank[0]+'-similarity_'+str(rank[1])[:7]))  
                    for rankIdx, rank in enumerate (duplicate_info_info)] 
                
                [print('@@@@@@\n',' qid: '+ str(idx)+'\nvId_fId: ' + str(duplicate_if[0])+' \nduplicate: '+ str(duplicate_if[1]))
                    for rankIdx, duplicate_if in enumerate (duplicate_info_result)] 
            #rank[1] : graph / rank[0] : graph db_idx
            
            # retreival_time = time.time() - retreival_start_time
            # print("@@@@@@@@@@@@@@@@@retreival_time@@@@@@@@@@@@@@@@@ :", retreival_time)

def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)


if __name__ == "__main__":
    vec1 = torch.tensor([[-0.014494836330414, -0.005165468901396, -0.046724624931812,
         -0.020842418074608,  0.039652168750763,  0.094548016786575,
          0.027805425226688, -0.070029020309448, -0.162854611873627,
         -0.007172767072916, -0.081028416752815, -0.104311451315880,
          0.033131036907434,  0.035504102706909, -0.129747629165649,
         -0.012566989287734, -0.064776763319969, -0.069441460072994,
         -0.011998843401670, -0.037030465900898, -0.148310482501984,
          0.005946309305727,  0.020134381949902, -0.096841149032116,
         -0.015379942953587, -0.055233411490917,  0.094897776842117,
         -0.005104694515467,  0.183701992034912,  0.120099067687988,
         -0.044839210808277, -0.060650080442429]])
    vec2 = torch.tensor([[-0.014494836330414, -0.005165468901396, -0.046724624931812,
         -0.020842418074608,  0.039652168750763,  0.094548016786575,
          0.027805425226688, -0.070029020309448, -0.162854611873627,
         -0.007172767072916, -0.081028416752815, -0.104311451315880,
          0.033131036907434,  0.035504102706909, -0.129747629165649,
         -0.012566989287734, -0.064776763319969, -0.069441460072994,
         -0.011998843401670, -0.037030465900898, -0.148310482501984,
          0.005946309305727,  0.020134381949902, -0.096841149032116,
         -0.015379942953587, -0.055233411490917,  0.094897776842117,
         -0.005104694515467,  0.183701992034912,  0.120099067687988,
         -0.044839210808277, -0.060650080442429]])
    
    sim = torch.sum(vec1 * vec2, dim=1).to(utils.get_device()) 
    print("sim: ",sim)
    cos = nn.CosineSimilarity(dim = 1)
    print("cos: ", cos(vec1, vec2))
    
    vec2 = torch.tensor([[-0.014494836330414, -0.005165468901396, -0.046724624931812,
    -0.020842418074608,  0.039652168750763,  0.094548016786575,
    0.027805425226688, -0.070029020309448, -0.162854611873627,
    -0.007172767072916, -0.081028416752815, -0.104311451315880,
    0.033131036907434,  0.035504102706909, -0.129747629165649,
    -0.012566989287734, -0.064776763319969, -0.069441460072994,
    -0.011998843401670, -0.037030465900898, -0.148310482501984,
    0.005946309305727,  0.020134381949902, -0.096841149032116,
    -0.015379942953587, -0.055233411490917,  0.094897776842117,
    -0.005104694515467,  0.183701992034912,  0.094897776842117,
    -0.044839210808277, -0.060650080442429]])
    
    sim = torch.sum(vec1 * vec2, dim=1).to(utils.get_device()) 
    print("sim: ",sim)
    cos = nn.CosineSimilarity(dim = 1)
    print("cos: ", cos(vec1, vec2))
    
    vec3 = torch.tensor([[-0.012270055711269, -0.004100099205971, -0.050724439322948,
         -0.012568246573210,  0.044939838349819,  0.095546767115593,
          0.032925292849541, -0.064932473003864, -0.163223624229431,
         -0.009922573342919, -0.076845899224281, -0.102719992399216,
          0.029639951884747,  0.034212484955788, -0.128898113965988,
         -0.016621567308903, -0.063070014119148, -0.067925751209259,
         -0.005379557609558, -0.030336214229465, -0.141939193010330,
          0.013977487571537,  0.021103031933308, -0.103314884006977,
         -0.019831305369735, -0.056288234889507,  0.095101520419121,
         -0.000310614705086,  0.190362334251404,  0.114728808403015,
         -0.042004764080048, -0.060694187879562]])
    
    
    sim = torch.sum(vec1 * vec3, dim=1).to(utils.get_device()) 
    print("sim: ",sim)
    cos = nn.CosineSimilarity(dim = 1)
    print("cos: ", cos(vec1, vec3))
    
    sys.exit()
    
    
    
    
    torch.set_printoptions(precision=20)
    main()

