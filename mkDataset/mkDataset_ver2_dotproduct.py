'''
  12.04  
  subgraph 기반의 TaGSim 이용 데이터 생성
  
  동일 비디오 - 동일 scenegraph 가 하나의 파일에 있음
  
  24.01.13.
  - 기존 mkDataset ver1은 distance, angle로 edge feature를 가졌음
  - version 2 는 distancae, angle 이 아닌 dotproduct 값만 갖도록 변경
  
'''
import sys, os
import pickle
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import networkx as nx
import multiprocessing as mp

from surel_gacc import run_walk
from utils.mkGraphRPE import *
import random
import math
import os

import pickle
import multiprocessing
#R_BFS 로 서브 그래프 생성
from itertools import combinations



def make_subgraph(graph, max_node, train, R_BFS):
    def split(node, subs, max, train, R_BFS, sub=None):
        if train:
            s = max//2
            max = random.randrange(s, max)

        if sub == None:
            sub = [node]

        cur = node
        while True:
            neig = list(graph.neighbors(cur))
            neig = list(set(neig)-set(sub))
            space = max-len(sub)
            if len(neig) == 0:
                # 더이상 갈 곳이 없는 경우
                sub.sort()
                subs.add(tuple(sub))
                break
            elif len(neig) <= space:
                # 여러 곳으로 갈 수 있을 경우
                if len(neig) == 1:
                    sub.extend(neig)
                    cur = neig[0]
                else:
                    sub.extend(neig)
                    if len(neig) == space:
                        sub.sort()
                        subs.add(tuple(sub))
                        break
                    if not R_BFS:
                        # 모든 상황 고려
                        for i in neig:
                            cur = i
                            tmp = sub.copy()
                            split(cur, subs, max, False, R_BFS, tmp)
                        break
                    else:
                        cur = random.choice(neig)
            else:
                # 갈 곳이 많지만 subgraph 노드 개수를 넘을 경우
                if not R_BFS:
                    for c in combinations(list(neig), space):
                        tmp = sub.copy()
                        tmp.extend(list(c))
                        tmp.sort()
                        subs.add(tuple(tmp))
                    break
                else:
                    # 교집합 부분으로 수정해야함
                    sub.extend(
                        list(random.choice(list(combinations(neig, space)))))
                    sub.sort()
                    subs.add(tuple(sub))
                    break

    subgraphs = []
    class_set = set()
    total_subs = set()
    for i in graph.nodes():
        split(i, total_subs, max_node, train, R_BFS)
    pre = [graph.subgraph(i) for i in total_subs]
    # 노드 클래스 중복으로 가지는 subgraph filtering
    for j in pre:
        class_sub = tuple([f['name'] for _, f in list(j.nodes.data())])
        if len(set(class_sub)) == 1:
            # continue
            # print(class_sub)
            break
        elif class_sub not in class_set:
            subgraphs.append(j)
            class_set.add(class_sub)
            class_set.add(tuple(reversed(class_sub)))

    return subgraphs


def mkMergeGraph(S, K, gT, nodeNameDict, F0dict, nodeIDDict):
    merged_K = np.concatenate([np.asarray(k) for k in K]).tolist()
    # print("merged_K: ",merged_K)
    merged_K = [nodeIDDict[i] for i in merged_K]
    # print("after merged_K: ", merged_K)
    
    sum_dict = {}
    count_dict = {}
    for k, gf in zip(merged_K, gT):
        if k in sum_dict:
            sum_dict[k] += gf
            count_dict[k] += 1
        else:
            sum_dict[k] = gf
            count_dict[k] = 1 
    gT_mean = {k: sum_dict[k] / count_dict[k] for k in sum_dict}
    
    return gT_mean 


def mkNG2Subs(G, args, F0dict):
    # print("---- mkNG2Subs ----")
    nmDict = dict((int(x), y['name'] ) for x, y in G.nodes(data=True)) 
    Gnode = list(G.nodes())  
    G_full = csr_matrix(nx.to_scipy_sparse_array(G))

    ptr = G_full.indptr
    neighs = G_full.indices
    num_pos, num_seed, num_cand = len(set(neighs)), 100, 5
    candidates = G_full.getnnz(axis=1).argsort()[-num_seed:][::-1]
    # print("candidates : ", candidates)
    rw_dict = {}
    B_queues  = []

    batchIdx, patience = 0, 0
    pools = np.copy(candidates)

    np.random.shuffle(B_queues)
    B_queues.append(sorted(run_sample(ptr,  neighs, pools, thld=1500)))
    B_pos = B_queues[batchIdx]

    B_w = [b for b in B_pos if b not in rw_dict]
    if len(B_w) > 0:
        walk_set, freqs = run_walk(ptr, neighs, B_w, num_walks=args.num_walks, num_steps=args.num_steps - 1, replacement=True)
    node_id, node_freq = freqs[:, 0], freqs[:, 1]
    rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))
    batchIdx += 1

    S, K, F = zip(*itemgetter(*B_pos)(rw_dict))

    F = np.concatenate(F)
    mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F])) 
    gT = mkGutils.normalization(mF, args)

    listA = [a.flatten().tolist() for a in K] 
    flatten_listA = list(itertools.chain(*listA)) 

    gT_concatenated = torch.cat((gT, gT), axis=1)
    enc_agg = torch.mean(gT_concatenated, dim=0)

    nodeIDDict = dict(zip(candidates, Gnode))
    rpeDict = mkMergeGraph (S, K, gT, nmDict, F0dict, nodeIDDict)
    for nodeId in list(G.nodes()):
        G.nodes()[nodeId]['rpe']  = rpeDict[nodeId] 

    return G, enc_agg
 

# GEV에 따른 TG 생성
# tombstone 기준으로 tsg_operations 간의 충돌을 확인하고, 충돌 없는 것 끼리 merge
#def mkTargetGraph(tsg_operations, tombstone, SourceGraph)
# return TG, TG에 사용된 tsg_operations, TG에 사용된 TSG
#nc -> change
#in -> insert

#GEV 생성 & TSG 생성
def mkTSGGenerator(graph, F0Dict, PredictDict, total_ged=0, max_node_idx=10):
    new_g = deepcopy(graph)
    new_g = nx.Graph(new_g)
    
    
    global_labels = list(F0Dict.keys()) 
    global_edge_labels = list(PredictDict.keys())


    target_ged = {}
    tombstone = []
    
    while (True):
        target_ged['nc'] = np.random.randint(0, max(int(new_g.number_of_nodes() / 3), 1))
        target_ged['ec'] = np.random.randint(0, max(int(new_g.number_of_edges() / 3), 1))
        target_ged['in'] = np.random.randint(1, 5)
        # target_ged['in'] = np.random.randint(max_node_idx, max_node_idx + 5)
        max_add_edge = min(int(((new_g.number_of_nodes() + target_ged['in']) * (
                    new_g.number_of_nodes() + target_ged['in'] - 1) / 2 - new_g.number_of_edges()) / 2), 4)
        if (max_add_edge <= (target_ged['in'] + 1)):
            max_add_edge = target_ged['in'] + 1
        target_ged['ie'] = np.random.randint(target_ged['in'], max_add_edge)

        temp_ged = target_ged['nc'] + target_ged['in'] + target_ged['ie'] + target_ged['ec']
        if (temp_ged != 0):
            break

    target_ged['nc'] = round(target_ged['nc'] * total_ged / temp_ged)
    target_ged['in'] = round(target_ged['in'] * total_ged / temp_ged)
    target_ged['ie'] = round(target_ged['ie'] * total_ged / temp_ged)
    target_ged['ec'] = round(target_ged['ec'] * total_ged / temp_ged)
    if (target_ged['ie'] < target_ged['in']):
        target_ged['ie'] = target_ged['in']

    #target_ged에는 각 operation이 적용되어 있음
#node label change - 실행
    to_edit_idx_newg = random.sample(new_g.nodes(), target_ged['nc'])
    # 새로운 그래프의 노드에서 target_ged['nc'] 개를 추출하여 to_edit_idx_newg로 생성
    for idx in to_edit_idx_newg:
        while (True):
            toassigned_new_nodetype = random.choice(list(global_labels))
            if (toassigned_new_nodetype != new_g.nodes()[idx]['name']):
                # tombstone['node'][idx] = {'operation': 'nc', 'new_label': toassigned_new_nodetype}
                tombstone.append({'operation': 'nc', 'node_id': idx, 'new_label': toassigned_new_nodetype})
                new_g.nodes()[idx]['name'] = toassigned_new_nodetype
                break
        
    if ((target_ged['ie'] - target_ged['in']) == 0):
        to_ins, to_del = 0, 0
    else:
        #node insertion 시 edge도 같이 추가하기 때문에(이때, 연결되는 노드는 랜덤)
        to_del = min(int(new_g.number_of_edges() / 3), np.random.randint(0, (target_ged['ie'] - target_ged['in'])))
        to_ins = target_ged['ie'] - target_ged['in'] - to_del

    deleted_edges = []
    for num in range(to_del):
        curr_num_egde = new_g.number_of_edges()
        to_del_edge = random.sample(new_g.edges(), 1)        
        #edge remove
        tombstone.append({'operation': 'ie', 'edge_tuple': (to_del_edge[0]),
                        'new_predicate': new_g.edges[to_del_edge[0][0], to_del_edge[0][1]]['predicate']})
        deleted_edges.append(to_del_edge[0])
        deleted_edges.append((to_del_edge[0][1], to_del_edge[0][0])) #양방향 그래프
        new_g.remove_edges_from(to_del_edge)
        
        assert ((curr_num_egde - new_g.number_of_edges()) == 1)

## edit edge labels
    if target_ged['ec'] !=0: 
        to_edit_idx_edge = random.sample(new_g.edges(), target_ged['ec'])
        
        for idx in to_edit_idx_edge:
            while (True):
                toassigned_new_edgetype = random.choice(global_edge_labels)
                
                if toassigned_new_edgetype != new_g.edges[idx[0], idx[1]]['predicate']:
                    break
            tombstone.append({'operation': 'ec', 'edge_tuple': (idx[0], idx[1]), 
                                'new_predicate': toassigned_new_edgetype})
            new_g.edges()[idx]['predicate'] = toassigned_new_edgetype  # 엣지의 레이블 변경

    ## edit node insertions
    for num in range(target_ged['in']):
        # curr_num_node = new_g.number_of_nodes()  #기존 그래프의 node 중 특정 node를 선택해 edge를 추가함
        #subgraph이므로 전체 노드 개수와 비교하여 생성해야함.
        # scenegraph의 maxnode 수 + target_ged operation의 개수+ 서브 그래프의 현재 노드 개수 + random.int(0,7)
        curr_num_node = (new_g.number_of_nodes() + max_node_idx+num+1 + random.randint(0,7))
        # print("curr_num_node : ", curr_num_node)
        to_insert_edge = random.sample(new_g.nodes(), 1)[0]
        label_name = random.choice(global_labels)

# node 추가
        new_g.add_node(curr_num_node, 
                       name=label_name,
                       bbox=  {'xmin': random.randint(0, 500), 'ymin': random.randint(0, 300), 
                               'xmax': random.randint(0, 500), 'ymax': random.randint(0, 300)},
                       txtemb = (F0Dict[label_name])
                       )
        if (curr_num_node !=to_insert_edge): 
            bbox_a = new_g.nodes[curr_num_node]['bbox']
            bbox_b = new_g.nodes[to_insert_edge]['bbox']
            center_a = ((bbox_a['xmin'] + bbox_a['xmax']) / 2, (bbox_a['ymin'] + bbox_a['ymax']) / 2)
            center_b = ((bbox_b['xmin'] + bbox_b['xmax']) / 2, (bbox_b['ymin'] + bbox_b['ymax']) / 2)
            
            dotproduct = center_a[0] * center_b[0] + center_a[1] * center_b[1]            
            predicate= random.choice(global_edge_labels)

# edge 추가
            new_g.add_edge(curr_num_node, to_insert_edge, 
                        predicate = predicate,
                        txtemb = PredictDict[predicate],
                        dotproduct = dotproduct
                        )
            
            tombstone.append({'operation': 'in', 'node_id': curr_num_node, 'new_label': label_name})
            tombstone.append({'operation': 'ie', 'edge_tuple': (curr_num_node, to_insert_edge), 'new_predicate': predicate})

    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        while (True):
            try:
                curr_pair = random.sample(new_g.nodes(), 2)
                bbox_a = new_g.nodes[curr_pair[0]]['bbox']
                bbox_b = new_g.nodes[curr_pair[1]]['bbox']

                center_a = ((bbox_a['xmin'] + bbox_a['xmax']) / 2, (bbox_a['ymin'] + bbox_a['ymax']) / 2)
                center_b = ((bbox_b['xmin'] + bbox_b['xmax']) / 2, (bbox_b['ymin'] + bbox_b['ymax']) / 2)

                dotproduct = center_a[0] * center_b[0] + center_a[1] * center_b[1]
                
                predicate=random.choice(global_edge_labels)
                if ((curr_pair[0], curr_pair[1]) not in deleted_edges):
                    if ((curr_pair[0], curr_pair[1]) not in new_g.edges()):                    
                        new_g.add_edge(curr_pair[0], curr_pair[1], name=random.choice(global_edge_labels),
                                    predicate = predicate,
                                    txtemb = PredictDict[predicate],
                                    dotproduct = dotproduct             
                        )
                        # tombstone['edge'][(curr_pair[0], curr_pair[1])] = {'operation': 'ie', 'new_predicate': predicate}
                        tombstone.append({'operation': 'ie', 'edge_tuple': (curr_pair[0], curr_pair[1]), 'new_predicate': predicate})  
                        break
                else:
                    break
            except:
                break
                
    return target_ged, new_g, tombstone

class TargetGraph:
    def __init__(self, original_scenegraph):
        self.graph = deepcopy(original_scenegraph)

    def apply_tsg_operation(self, tsg_operation, tsg):
        operation_type = tsg_operation['operation']
        if operation_type == 'nc':
            node_id = tsg_operation['node_id']
            new_label = tsg_operation['new_label']           
            self.apply_node_change(node_id, new_label, tsg)            
        elif operation_type == 'ec':
            edge_tuple = tsg_operation['edge_tuple']
            new_predicate = tsg_operation['new_predicate']
            self.apply_edge_change(edge_tuple, new_predicate, tsg)
        elif operation_type == 'in':
            node_id = tsg_operation['node_id']
            node_label = tsg_operation['new_label']
            self.apply_node_change(node_id, node_label, tsg)
        elif operation_type == 'ie':
            edge_tuple = tsg_operation['edge_tuple']
            new_predicate = tsg_operation['new_predicate']
            self.apply_edge_insertion(edge_tuple, new_predicate, tsg)

    def apply_node_change(self, node_id, new_label, tsg):
        # print("nc")
        node_attributes = tsg.nodes[node_id]
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(node_attributes)
        else:
            self.graph.add_node(node_id, **node_attributes)

    def apply_edge_change(self, edge_tuple, new_predicate, tsg):
        # print("ec")
        edge_attributes = tsg.edges[edge_tuple]
        if self.graph.has_edge(*edge_tuple):
            self.graph.edges[edge_tuple].update(edge_attributes)
        else:
            self.graph.add_edge(edge_tuple[0], edge_tuple[1], **edge_attributes)
    
    def apply_node_insertion(self, node_id, tsg):
        # print("in") # tsg에서 노드 속성 가져오기
        node_attributes = tsg.nodes[node_id] # 기존 그래프에 노드가 있는지 확인
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(node_attributes) # 기존 노드가 있다면 속성 업데이트
        else:
            self.graph.add_node(node_id, **node_attributes) # 기존 노드가 없다면 노드 추가
    
            

    def apply_edge_insertion(self, edge_tuple, new_predicate, tsg):
        # print("ie")
        if tsg.has_edge(*edge_tuple): # insertion
            edge_attributes = tsg.edges[edge_tuple] 
            if self.graph.has_edge(*edge_tuple): # 기존 엣지가 있다면 속성 업데이트
                self.graph.edges[edge_tuple].update(edge_attributes)
            else: #없으면 추가
                self.graph.add_edge(edge_tuple[0], edge_tuple[1], **edge_attributes)      
        else: # deletion
            if self.graph.has_edge(*edge_tuple):  # 있으면 삭제
                self.graph.remove_edge(*edge_tuple)
            else: # self.graph.has_edge(edge_tuple[1], edge_tuple[0]): #반대 방향으로 있을 수 있는 edge 삭제. 
                self.graph.remove_edge(edge_tuple[1], edge_tuple[0])
        
# 'node_index'와 'edge_tuple'을 추출하는 함수
def extract_labels_and_tuples(tombstone):
    node_labels = [op['new_label'] for op in tombstone if 'new_label' in op]
    edge_tuples = [op['edge_tuple'] for op in tombstone if 'edge_tuple' in op]
    return node_labels, edge_tuples

# 순차적으로 겹치는 부분이 있는지 탐색하는 함수
def sequential_overlap_search(tombstone_list):
    notDuplicatePair = []
    for i in range(len(tombstone_list)):
        for j in range(i + 1, len(tombstone_list)):
            labels_a, tuples_a = extract_labels_and_tuples(tombstone_list[i])
            labels_b, tuples_b = extract_labels_and_tuples(tombstone_list[j])
            if not set(labels_a).intersection(labels_b) and not set(tuples_a).intersection(tuples_b):
                notDuplicatePair.append((i, j))
                # print(f"No overlap between tombstone {i} and tombstone {j}")
    #  notDuplicatePair에 포함되지 않은 Index의 tombstone을 tIdx로 지정하고, (tIdx, tIdx)를 notDuplicatePair에 추가
    for tIdx in range(len(tombstone_list)):
        if all(tIdx not in pair for pair in notDuplicatePair):
            notDuplicatePair.append((tIdx, tIdx))  

    return notDuplicatePair


'''
    각 operation list간 겹치는 원소가 없는 것끼리 짝 찾기
'''

from collections import OrderedDict
def get_target_operations(no_duplication):
    results = []
    fin_results = []
    for i in range(len(no_duplication)):
        for j in range(i + 1, len(no_duplication)):
            cur_tsg = list(set(no_duplication[i] + no_duplication[j]))
            tg_oper = []
            for k in range(len(cur_tsg)):
                for l in range(k + 1, len(cur_tsg)):
                    tg_oper.append((cur_tsg[k], cur_tsg[l]))
            if all(tg_oper in no_duplication for tg_oper in tg_oper):
                results.append(tg_oper)
    for pair in results:
        cur_pair = []
        [cur_pair.extend(tpl) for tpl in pair]
        fin_results.append(list(set(cur_pair)))
    # print("중복제거 전 fin_results: ",fin_results)
    
    seen = set()
    after_fin_results = []
    for item in fin_results:
            if tuple(item) not in seen:
                seen.add(tuple(item))
                after_fin_results.append(item) 
     
    #no_duplication의 원소 중 after_fin_results의 원소에 하나도 포함되지 않은 조합을 추가
    # (두 개 tsg 로 구성된 TargetGraph 생성 Operation을 찾기 위함)           
    other_targetG = []
    for item in no_duplication:
        # found = any(any(elem in result for elem in item) for result in after_fin_results)
        found = False
        for result in after_fin_results:
            for elem in item:
                if elem in result:
                    found = True
                    break
            if found:
                break
        if not found:
            other_targetG.append(item)
    # print("other_targetG : ", other_targetG)
    # print("after_fin_results : ", after_fin_results)
    
    if len(other_targetG) != 0:
        [after_fin_results.append(list(pair)) for pair in other_targetG]
    
    return after_fin_results

def mkTargetGraph(original_scenegraph, tsg_operations_list, tsg_list, tombstone_list):
    target_graphs = []
    current_target_graph = None
    tsg4tg_list = []
    # print("tombstone_list: ",tombstone_list)
    #todo - 중첩되지 않는 쌍이 없는 경우(모든 tombstone_list가 동일할 경우, Target graph 가 생성되지 않음 - 노드가 3개인데, GED가 10인 경우?) 

    notDuplicatePair = sequential_overlap_search(tombstone_list)
    tsg4tg_list.extend(get_target_operations(notDuplicatePair)) # 원소 하나가 중복 제거된 tsg 집합
    # print("중복 제거 후:", tsg4tg_list)      
    # print("tsg_operations_list: ", tsg_operations_list)
    # 
    # print("notDuplicatePair: ",notDuplicatePair)
    # print("tsg4tg_list: ",tsg4tg_list)
       
    
    for tIdx, tsgIdxList in enumerate(tsg4tg_list): #TG 
        # print(" tsg_operations_list[tIdx]: ",  tsg_operations_list[tIdx])
        target_graph = TargetGraph(original_scenegraph)
        targetGED =  {'nc': 0, 'ec': 0, 'in': 0, 'ie': 0}
        # todo - tsgIdxList의 평균 개수, min, max 필요
        # todo - edge insertion에 동일 edge에 대해 insertion되는 경우를 보였음. 이에 대해 확인 후 처리 필요
        for tsgOperation in tsgIdxList: #Tsg 번호
            # 중복되지 않는 operation을 가진 tsg 들을 하나의 TargetGraph에 적용시킴
            # 기본 targetGraphClass에서 사용하는 method 에 tsg를 보내서, 동일한 노드를 복사해서 넣을 수 있도록 변경해야함(bbox나 txtembedding 값 때문에)
            cur_tsg = tsg_list[tsgOperation]
            for operation in tombstone_list[tsgOperation]:
                #operation에 맞는 tsg 를 argument로 넣어서 동일한 노드, edge를 복제하게 함
                target_graph.apply_tsg_operation(operation, cur_tsg)               
            #GED 추가
            for key, value in tsg_operations_list[tsgOperation].items():
                targetGED[key] += value
                
        tsg_graphs = [tsg_list[tsgOperation] for tsgOperation in  tsgIdxList]
        target_graphs.append([tIdx, target_graph.graph, targetGED, tsgIdxList, tsg_graphs]  ) 
        # target_graphs.append([tIdx, target_graph.graph, targetGED, tsgIdxList]  ) 
        #target_graphs.append(targetGraph Idx, TargetGraph, 전체 사용한 GEV , 사용한 tsg 그래프 Idx,  )
    return target_graphs



'''
    - 전체 장면 그래프 개수
    - 서브 그래프 생성 기준
    - 하나의 비디오에서 몇 장의 scenegraph가 있는지
        - 한 장면 그래프의 평균 노드 수
    - 하나의 scenegraph에서 몇 개의 subgraph가 있는지
        - 서브 그래프의 노드, edge 개수 확인
'''
def PairDataset(filenames, F0Dict,PredictDict,total_ged, train, args ):
    # file_path,train_num_per_row,  dataset, total_ged, train, args
    for filename in filenames:
        # scenegraph data load
        fpath = "data/dataset02/scenegraph_v2/"+str(filename)    
        with open(fpath, 'rb') as file:
          data = pickle.load(file)
        dataset = data[0] #video 내 graphs
        print("------- PairDataset ---------")
        
        length = len(dataset) # 비디오(scenegraph 묶음)
        
        if length != 0:
            print("tqdm - length: ", length)
            print("tqdm - filename: ", filename)
            
            SG_TG_list = []
            if train: 
                g1_list = []
                g2_list = []
                ged_list = [] 
                for i in range(length):
                    # print("scenegraph loop")
                    # print("train == True")     
                    # Source Graph 에 RPE 속성을 추가하고, SubGraph로 나눔
                    dataset[i].graph['gid'] = 0 # i로 올라가게 해야 하는 것 아님?
                    # originSourceGraph = dataset[i]
                    SourceGraph, enc_agg = mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 RPE Feature 붙임
                    #R_BFS(Random breath-firs search)로 서브 그래프 생성
                    
                    #todo 해당 subgraph 생성 방법 변경 필요 @@@@@@@@@@@@@@
                    
                    
                    sourceSubGraphs = make_subgraph(SourceGraph, 4, False, False)                    
                    
                    # print("sourceSubGraphs: ", sourceSubGraphs)
                    tsg_list = [] #syntheic graph from subgraph
                    tsg_operations_list = [] #GEV from provide GED   
                    tombstone_list = []
                    
                    max_node_idx = len(dataset[i].nodes)            
                    # print("max_node_idx: ",max_node_idx)
                    
                    #scenegraph당 생성? video 당 생성.
                    for ssgIdx, SSG in enumerate(sourceSubGraphs):                        
                        #total_ged를 기반으로 operations를 생성하고, 각 operation을 해당 subgraph에 적용
                        tsg_operations, TSG, tombstone = mkTSGGenerator(SSG, F0Dict, PredictDict, total_ged, max_node_idx)
                        # 모든 tsg_operations, TSG, tombstone을 바탕으로 TargetGraph를 생성해야 하므로, 일단 SourceGraph에 대응하는 targetGraph를 모두 모아둠      
                        tsg_list.append(TSG)
                        tsg_operations_list.append(tsg_operations)
                        tombstone_list.append(tombstone)                      
                    # RPE가 없는 그래프에 충돌하지 않는 tsg_operation들을 적용하여 TargetGraph를 생성
                    #target grpah의 조건에 맞게  target graph(scene graph 생성)
                    TG_list = mkTargetGraph(dataset[i], tsg_operations_list, tsg_list, tombstone_list)
                    SG_TG_list.append([SourceGraph, TG_list]) # Source Graph와 대응되는 TargetGraph 의 List - metadata
                                       
                    
                    for TGidx, tg_ in enumerate(TG_list) : 
                        # print("TG_list loop")
                        #text emb 값 할당
                        TG_rpe, enc_agg = mkNG2Subs(tg_[1], args, F0Dict)  # Gs에 Feature 붙임

                        for tsgIdx in tg_[3]:
                            cur_tsg = deepcopy(tsg_list[tsgIdx])
                            for nodeIdx in cur_tsg.nodes():
                                node_attributes = TG_rpe.nodes[nodeIdx]
                                cur_tsg.nodes[nodeIdx].update(node_attributes)
                            #print("cur_tsg: ",cur_tsg.nodes()) # meta - subgraph의 평균 노드 개수
                            # print(sourceSubGraphs[tsgIdx], cur_tsg, tsg_operations_list[tsgIdx]) #기존 데이터셋 규격
                        #meta dataset
                            g1_list.append(sourceSubGraphs[tsgIdx]) #source subgraph
                            g2_list.append(cur_tsg) #target subgraph
                            ged_list.append(list(tsg_operations_list[tsgIdx].values()))  # source subgraph to target subgraph - GEV({}X, []O)

                    # 완성된 dataset 저장
                    if i == length-1:
                    # if i == 1:    
                        with open("data/dataset02/meta_dataset02/walk4_step3_ged5/{}_{}.pickle".format(filename[:-9], len(SG_TG_list)), "wb") as fw:
                            pickle.dump(SG_TG_list, fw)       
                                                    
                        with open("data/dataset02/walk4_step3_ged5/walk{}_step{}_ged{}_{}_{}_{}.pickle".format(args.num_walks,args.num_steps,total_ged, filename[:-9], len(SG_TG_list), len(ged_list)), "wb") as f:
                            pickle.dump([g1_list, g2_list, ged_list], f, protocol=pickle.HIGHEST_PROTOCOL)
                        print("dump! - i: {} / filename: {}".format(i, filename)) # subgraph개수..? scenegraph 개수를 따로 해둬야 하나..?
                
        else :
            print("length is 0 -> killed")


def distribute_files_by_size(file_list, num_processes):
    # 파일 크기에 따라 파일 리스트를 정렬
    sorted_files = sorted(file_list, key=lambda f: os.path.getsize("data/dataset02/scenegraph_v2/" + f), reverse=True)

    # 정렬된 파일 리스트를 프로세스에 균등하게 분배
    split_filenames = [[] for _ in range(num_processes)]
    for i, file_name in enumerate(sorted_files):
        split_filenames[i % num_processes].append(file_name)
    return split_filenames

# 각 서버 간 processor 수가 안맞는 부분 해결 위해.. (동일한 비디오에 대해 수행하게 하려고.)
def split_list(lst, num_proc):
    avg = len(lst) // num_proc
    remainder = len(lst) % num_proc
    result = []

    start = 0
    for i in range(num_proc):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

def main(margs):
    # node class txtemb load <- mkScenegraph에서 만든 것
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
       data  = pickle.load(f)
    F0Dict = data
    
    # edge class txtemb load <- mkScenegraph에서 만든 것
    with open('data/predicate_unique_textemb.pickle', 'rb') as f:
        data  = pickle.load(f)
    PredictDict = data
    
    # 폴더 내 파일(Scenegraph) 목록 로드
    folderpath = "data/dataset02/scenegraph_v2"
    # file_list = os.listdir(folderpath)

    # @@@ temp
    # file_list = file_list[:5]
    train = True
    provisedGED = 5

    # # 파일 목록을 프로세스별로 분할
    num_processes = multiprocessing.cpu_count()
    
    with open('data/fileNameList_ordered.pkl', 'rb') as f:
        fileNameList  = pickle.load(f)
    resultList = []

    for item in fileNameList:
            sliced_item = item[-2:-1]
            # sliced_item = item[7:14]  
            # sliced_item = item[-13:-12]
            # sliced_item = item[-15:-14]
            resultList.extend(sliced_item)

    print(resultList[0])
    total_list = [file.split('.')[0] for file in resultList]
    
    # 2. '_'를 기준으로 슬라이싱하여 [3]을 원소로 하는 리스트 exist_list 생성    
    # folder_path = 'data/dataset01/walk4_step3_ged5/'
    folder_path = 'data/dataset02/walk4_step3_ged5/'
    file_names = os.listdir(folder_path)

    exist_list = [file.split('_')[3] for file in file_names]

    # 3. total_list의 원소 중 exist_list에 없는 원소들을 모아 unexi_list로 모음
    unexi_list = [item+'.pkl' for item in total_list if item not in exist_list]

    # 결과 출력
    print("Total List:", total_list)
    print("Exist List:", exist_list)
    print("Unexistent List:", unexi_list)
    print("Exist List:", len(exist_list))
    
    print("Total List:", total_list[0])
    print("Unexistent List:", len(unexi_list))
    print("Unexistent List:", unexi_list[0])
    
    regen_resultList = split_list(unexi_list, num_processes)

    # PairDataset (resultList[0], F0Dict, PredictDict, provisedGED, train, margs)
    # 프로세스를 생성하고 딕셔너리를 개별적으로 전달
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=PairDataset, args=(regen_resultList[i], F0Dict, PredictDict, provisedGED, train, margs ))
        process.start()
        processes.append(process)

    # # 모든 프로세스가 종료될 때까지 기다림
    for process in processes:
        process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    margs = parser.parse_args()
    main(margs)