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

from itertools import combinations
import networkx as nx


import matplotlib.pyplot as plt
import pickle
import random



'''
    subgraph 생성
    -> GEDPair 생성
    -> 학습

'''

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
    # 노드 클래스가 중복으로 가지는 subgraph filtering
    for j in pre:
        class_sub = tuple([f['name'] for _, f in list(j.nodes.data())])
        if len(set(class_sub)) == 1:
            continue
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
 

def graph_generation(graph, F0Dict, PredictDict, total_ged=0):
    new_g = deepcopy(graph)
    
    global_labels = list(F0Dict.keys()) 
    global_edge_labels = list(PredictDict.keys())

    target_ged = {}
    while (True):
        target_ged['nc'] = np.random.randint(0, max(int(new_g.number_of_nodes() / 3), 1))
        target_ged['ec'] = np.random.randint(0, max(int(new_g.number_of_edges() / 3), 1))
        target_ged['in'] = np.random.randint(1, 5)
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


    to_edit_idx_newg = random.sample(new_g.nodes(), target_ged['nc'])
    for idx in to_edit_idx_newg:
        while (True):
            toassigned_new_nodetype = random.choice(list(global_labels))
            if (toassigned_new_nodetype != new_g.nodes()[idx]['name']):
                break
        new_g.nodes()[idx]['name'] = toassigned_new_nodetype

    if ((target_ged['ie'] - target_ged['in']) == 0):
        to_ins, to_del = 0, 0
    else:
        to_del = min(int(new_g.number_of_edges() / 3), np.random.randint(0, (target_ged['ie'] - target_ged['in'])))
        to_ins = target_ged['ie'] - target_ged['in'] - to_del

    deleted_edges = []
    for num in range(to_del):
        curr_num_egde = new_g.number_of_edges()
        to_del_edge = random.sample(new_g.edges(), 1)
        deleted_edges.append(to_del_edge[0])
        deleted_edges.append((to_del_edge[0][1], to_del_edge[0][0]))
        new_g = nx.Graph(new_g)
        new_g.remove_edges_from(to_del_edge)
        assert ((curr_num_egde - new_g.number_of_edges()) == 1)

    ## edit edge labels
    to_edit_idx_edge = random.sample(new_g.edges(), target_ged['ec'])
    for idx in to_edit_idx_edge:
        while (True):
            toassigned_new_edgetype = random.choice(global_edge_labels)
            if (toassigned_new_edgetype != new_g.edges()[idx]['predicate']):
                break
        new_g.edges()[idx]['predicate'] = toassigned_new_edgetype
    
    ## edit node insertions
    for num in range(target_ged['in']):
        curr_num_node = new_g.number_of_nodes()
        to_insert_edge = random.sample(new_g.nodes(), 1)[0]
        label_name = random.choice(global_labels)
        # new_g = deepcopy(new_g)
        new_g = nx.Graph(new_g) # unfrozen. 위에서 copy 했는데도 frozen임..왜..?
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

            distance = math.sqrt((center_b[0] - center_a[0])**2 + (center_b[1] - center_a[1])**2)

            deltaX_AB = center_b[0] - center_a[0]
            deltaY_AB = center_b[1] - center_a[1]
            angle_AB = math.degrees(math.atan2(deltaY_AB, deltaX_AB))

            deltaX_BA = center_a[0] - center_b[0]
            deltaY_BA = center_a[1] - center_b[1]
            angle_BA = math.degrees(math.atan2(deltaY_BA, deltaX_BA))

            predicate= random.choice(global_edge_labels)
            new_g = nx.Graph(new_g)
            new_g.add_edge(curr_num_node, to_insert_edge, 
                        predicate = predicate,
                        txtemb = PredictDict[predicate],
                        distance= distance, angle_AB = angle_AB,
                        angle_BA = angle_BA
                        )
    
    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        # while (True):
        try:
            curr_pair = random.sample(new_g.nodes(), 2)

            bbox_a = new_g.nodes[curr_pair[0]]['bbox']
            bbox_b = new_g.nodes[curr_pair[1]]['bbox']

            center_a = ((bbox_a['xmin'] + bbox_a['xmax']) / 2, (bbox_a['ymin'] + bbox_a['ymax']) / 2)
            center_b = ((bbox_b['xmin'] + bbox_b['xmax']) / 2, (bbox_b['ymin'] + bbox_b['ymax']) / 2)

            # A와 B의 거리 계산
            distance = math.sqrt((center_b[0] - center_a[0])**2 + (center_b[1] - center_a[1])**2)
            # 객체 A를 기준으로 객체 B의 상대 각도 계산
            deltaX_AB = center_b[0] - center_a[0]
            deltaY_AB = center_b[1] - center_a[1]
            angle_AB = math.degrees(math.atan2(deltaY_AB, deltaX_AB))
            # 객체 B를 기준으로 객체 A의 상대 각도 계산
            deltaX_BA = center_a[0] - center_b[0]
            deltaY_BA = center_a[1] - center_b[1]
            angle_BA = math.degrees(math.atan2(deltaY_BA, deltaX_BA))

            predicate=random.choice(global_edge_labels)
            if ((curr_pair[0], curr_pair[1]) not in deleted_edges):
                if ((curr_pair[0], curr_pair[1]) not in new_g.edges()):                    
                    new_g.add_edge(curr_pair[0], curr_pair[1], name=random.choice(global_edge_labels),
                                predicate = predicate,
                                txtemb = PredictDict[predicate],
                                distance= distance, 
                                angle_AB = angle_AB,
                                angle_BA = angle_BA                               
                    )
                    break
            else:
                break
        except:
            print("EXCEPT")
            continue
    
    return target_ged, new_g


def PairDataset(filenames, F0Dict,PredictDict,total_ged, train, args ):
    train_num_per_row = 64
    # file_path,train_num_per_row,  dataset, total_ged, train, args
    for filename in filenames:
        # 파일 처리 작업
        # print("filename: ", filename)
        fpath = "data/scenegraph/"+str(filename)    
        with open(fpath, 'rb') as file:
          data = pickle.load(file)
        dataset = data[0] #video 내 graphs
# with open("data/networkx_ver3_100000/v3_x1000.pickle", "rb") as fr:
#         dataset = pickle.load(fr)
        total = []
        for i in range(len(dataset)):
            if train:
                subs = make_subgraph(dataset[i], 4, False, False)
                total.extend(subs)
        
        print("------- PairDataset ---------")
        g1_list = []
        g2_list = []
        ged_list = []
        
        # length = len(dataset)
        length = len(total)
        # cnt = 0
        if length != 0:
            print("tqdm - length: ", length)
            print("tqdm - filename: ", filename)
            cnt = 0
            for i in range(length):
                if train:                    
                    # try: 
                        total[i].graph['gid'] = 0
                        #text emb 값
                        target_ged, new_g = graph_generation(total[i], F0Dict, PredictDict, total_ged)

                        new_g, enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
                        origin_g, enc_agg = mkNG2Subs(total[i], args, F0Dict)  # Gs에 Feature 붙임
                        graph2 = new_g

                        gev = [target_ged['nc'],target_ged['ec'],target_ged['in'],target_ged['ie'],]
                        graph2.graph['gid'] = 1

                        g1_list.append(origin_g)
                        g2_list.append(graph2)
                        ged_list.append(gev)  
                    # except:
                    #     print("ERR - total[i]: {}".format(total[i]))
                    #     print("ERR - len(total[i]): {}".format(len(total[i])))
                    #     continue

                # try:
                if i == length-1:
                    with open("data/train_sub-16-13/walk4_step3_ged10/walk{}_step{}_ged{}_{}_{}.pkl".format(args.num_walks,args.num_steps,total_ged, filename[:-9], i), "wb") as fw:
                        pickle.dump([g1_list, g2_list, ged_list], fw)
                    print("dump! - i: {} / filename: {}".format(i,filename))
                    g1_list = []
                    g2_list = []
                    ged_list = []

                elif cnt == 100:
                    with open("data/train_sub-16-13/walk4_step3_ged10/walk{}_step{}_ged{}_{}_{}.pkl".format(args.num_walks,args.num_steps,total_ged, filename[:-9], i),  "wb") as fw:
                        pickle.dump([g1_list, g2_list, ged_list], fw)
                    print("dump! - i: {} / filename: {} / cnt: {}".format(i, filename, cnt))

                    g1_list = []
                    g2_list = []
                    ged_list = []

                    cnt = 0
                else:
                    cnt += 1
                # except:
                #     print("ERR - dump")
                #     continue
        else :
            print("length is 0 -> killed")


def distribute_files_by_size(file_list, num_processes):
    # 파일 크기에 따라 파일 리스트를 정렬
    sorted_files = sorted(file_list, key=lambda f: os.path.getsize("data/scenegraph/" + f), reverse=True)

    # 정렬된 파일 리스트를 프로세스에 균등하게 분배
    split_filenames = [[] for _ in range(num_processes)]
    for i, file_name in enumerate(sorted_files):
        split_filenames[i % num_processes].append(file_name)

    return split_filenames


def main(margs):
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
       data  = pickle.load(f)
    F0Dict = data

    with open('data/predicate_unique_textemb.pickle', 'rb') as f:
        data  = pickle.load(f)
    PredictDict = data

    # 폴더 내 파일 목록 로드
    folderpath = "data/scenegraph"
    file_list = os.listdir(folderpath)

    # file_list = file_list[:5]
    train = True
    total_ged = 10


    # # 파일 목록을 프로세스별로 분할
    num_processes = multiprocessing.cpu_count()
    
    with open('data/fileNameList_ordered.pkl', 'rb') as f:
        fileNameList  = pickle.load(f)
    resultList = []
    for item in fileNameList:
        sliced_item = item[-16:-13] 
        # sliced_item = item[-13:-10]
        resultList.append(sliced_item)

    # 프로세스를 생성하고 딕셔너리를 개별적으로 전달
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=PairDataset, args=(resultList[i], F0Dict, PredictDict, total_ged, train, margs ))
        process.start()
        processes.append(process)

    # 모든 프로세스가 종료될 때까지 기다림
    for process in processes:
        process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    margs = parser.parse_args()
    main(margs)

   
