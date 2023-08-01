import os
import pickle
import threading

import pickle
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import networkx as nx

from surel_gacc import run_walk
from utils.mkGraphRPE import *
import random
import math
from tqdm import tqdm


def check_deadlock():
    # 현재 실행 중인 스레드 확인
    current_thread = threading.current_thread()

    # 모든 스레드 확인
    all_threads = threading.enumerate()

    # 모든 스레드의 상태와 락 상태 출력
    for thread in all_threads:
        print(f"Thread name: {thread.name}, is alive: {thread.is_alive()}, lock status: {thread.locked()}")
        
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



'''
    S의 각 원소(워크셋)를 이용해 edge list를 생성하고, edge에 따라 node를 생성하는 방법으로 하나의 서브그래프로 병합
'''
# path 로 edge list 만들고 edge 추가하기; node path로 Graph 생성 
def mkMergeGraph(S, K, gT, nodeNameDict, F0dict, nodeIDDict):
    # print("-------mkMergeGraph-------")
    # --------- vvvvv 생성된 walk에 있는 모든 노드들을 각 노드의 id에 맞게 rpe 를 concat 후 mean pooling 한 값 ----------------------
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
    # print('sum_dict.keys' ,sum_dict.keys)
    # print('sum_dict.values' ,sum_dict.values)
    # print('gT_mean:' ,gT_mean)
    
    return gT_mean 

'''
    새로운 그래프 new_에 대해 run_walk를 이용해 rpe enc값을 추출하고, 특징값을 concat + 하나의 서브 그래프로 병합 -> 과정에서 불필요한 값이 추가됨; 이를 확인 필요
     -> 해당 subgraph에 대한 rpe embedding값을 얻을 수 있고, 새로 생성해서 사라진 name과 node idx 값 외의 속성을 추가(F0, rpe값)
    특징값 concat 할 때, 기존과 동일하도록 concat 필요
'''
# def mkNG2Subs(G, args, F0dict, originGDict):
def mkNG2Subs(G, args, F0dict):
    # originGraph의 feature를 가져옴
    nmDict = dict((int(x), y['name'] ) for x, y in G.nodes(data=True)) # id : name 값인 Node Name Dict
    Gnode = list(G.nodes())   # Gnode의 각 원소와 pools의 원소가 key-value가 되게 매칭,  -> item getter에 맞게 변경 해야함
    # print("Gnode : ",Gnode)
    
    # G_full = nx2csr(G) # mkGraphRPE에 있는 함수, return csr_matrix(nx.to_scipy_sparse_array(G))
    G_full = csr_matrix(nx.to_scipy_sparse_array(G))
    # print("G_full: ", G_full) #각 노드간 연결을 표현할 뿐, 그 외는 표현 불가능

    ptr = G_full.indptr
    neighs = G_full.indices
    num_pos, num_seed, num_cand = len(set(neighs)), 100, 5
    candidates = G_full.getnnz(axis=1).argsort()[-num_seed:][::-1]
    # print("candidates : ", candidates)
    rw_dict = {}
    B_queues  = []

    # for r in range(1, args.repeat + 1): # 모든 노드에 대해 한 번씩 할거라 repeat 필요 X
    batchIdx, patience = 0, 0
    pools = np.copy(candidates)

    # while True:
    np.random.shuffle(B_queues)
    # if r <= 1:
    B_queues.append(sorted(run_sample(ptr,  neighs, pools, thld=1500))) # pool를 인자로 넣어 모든 노드에 대해 수행하도록 함

    B_pos = B_queues[batchIdx]

    B_w = [b for b in B_pos if b not in rw_dict]
    if len(B_w) > 0:
        walk_set, freqs = run_walk(ptr, neighs, B_w, num_walks=args.num_walks, num_steps=args.num_steps - 1, replacement=True)
    node_id, node_freq = freqs[:, 0], freqs[:, 1]
    rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))
    batchIdx += 1

    # obtain set of walks, node id and DE (counts) from the dictionary
    S, K, F = zip(*itemgetter(*B_pos)(rw_dict))

    # print("S: ",S[0]): 0번 노드로 시작하는 서브 그래프( walks set ) -> len(walks set): num_walks*(num_step+1)
    # print("K: ",K[0]): 0번 노드로 시작하는 서브 그래프의 .nodes() <- 이때 해당 노드의 F0 값은 없음
    # print("F: ",F[0]): S[0]의 Feature 값; 

    F = np.concatenate(F) #([[[0] * F.shape[-1]], F])   # rpe encoding 값(step 들만)
    mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F]))  #.to(device) # walk를 각 노드에 맞춰서 concat
    gT = mkGutils.normalization(mF, args)

    listA = [a.flatten().tolist() for a in K] 
    flatten_listA = list(itertools.chain(*listA))  # 35*12

    gT_concatenated = torch.cat((gT, gT), axis=1)
    enc_agg = torch.mean(gT_concatenated, dim=0) # todo 서브 그래프 feature 값...  다 concat.. <.,..?졸려서 머리가 안돌아감. 생각 보류. 

    nodeIDDict = dict(zip(candidates, Gnode)) # sampling하고 나면 원래의 id가 아니게 됨.. 순서 안변함.
    rpeDict = mkMergeGraph (S, K, gT, nmDict, F0dict, nodeIDDict)
    for nodeId in list(G.nodes()):
        G.nodes()[nodeId]['rpe']  = rpeDict[nodeId]

    ''' 
    listA = [a.flatten().tolist() for a in K] 
    flatten_listA = list(itertools.chain(*listA))  # 35*12
    print(len(flatten_listA))
    print("K의 노드 개수: ", len(flatten_listA))
        -> K에는 중복되는 노드 id가 있지만, 각 노드는 서브그래프(S[i]) 내에서 다른 RPE값을 가짐(F[i], mF[i])

        때문에 이를 토대로 서브 그래프 생성(mkPathGraph) 후 해당 RPE 값을  attribute로 부여해줘야하고, 
        또 맨 위에서 만들어 둔 nodeDict를 이용해서 name에 따른 feature 값인 F0값을 부여해줘야함      
        '''
    return G, enc_agg
 

'''
    targer GEV 를 랜덤으로 생성하고 그에 맞는 pair graph를 생성
@@@
    bbox는 임의의 값으로 지정함

'''
def graph_generation(graph, F0Dict, PredictDict, total_ged=0):
    new_g = deepcopy(graph)
    
    global_labels = list(F0Dict.keys()) # name / value 로 txtemb 값이 들어가야 함
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
    # print('target ged :', target_ged)

    ## edit node labels
    to_edit_idx_newg = random.sample(new_g.nodes(), target_ged['nc'])
    for idx in to_edit_idx_newg:
        while (True):
            toassigned_new_nodetype = random.choice(list(global_labels))
            if (toassigned_new_nodetype != new_g.nodes()[idx]['name']):
                break
        new_g.nodes()[idx]['name'] = toassigned_new_nodetype

    ## edit edge deletion
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
        new_g.remove_edges_from(to_del_edge)
        assert ((curr_num_egde - new_g.number_of_edges()) == 1)

    ## edit edge labels
    to_edit_idx_edge = random.sample(new_g.edges(), target_ged['ec'])
    for idx in to_edit_idx_edge:
        while (True):
            toassigned_new_edgetype = random.choice(global_edge_labels)
            if (toassigned_new_edgetype != new_g.edges()[idx]['predicate']):
                break
                # print('line 210 -  edge name')
        new_g.edges()[idx]['predicate'] = toassigned_new_edgetype
    
    ## edit node insertions
    for num in range(target_ged['in']):
        curr_num_node = new_g.number_of_nodes()
        to_insert_edge = random.sample(new_g.nodes(), 1)[0]
    #bbox 추가, feature emb 추가, edge feature 계산
        # new_g.add_node(str(curr_num_node), label=str(curr_num_node), name=random.choice(global_labels))
        label_name = random.choice(global_labels)
        new_g.add_node(curr_num_node, 
                       name=label_name,
                       bbox=  {'xmin': random.randint(0, 500), 'ymin': random.randint(0, 300), 
                               'xmax': random.randint(0, 500), 'ymax': random.randint(0, 300)},
                       txtemb = (F0Dict[label_name])
                       )
        # print("edge feature cal")
        
        if (curr_num_node !=to_insert_edge): 
            #mkSceneGraph에서 사용한 것 할 수 없음 - edge feature 사용
            bbox_a = new_g.nodes[curr_num_node]['bbox']
            bbox_b = new_g.nodes[to_insert_edge]['bbox']

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
            # add edge to the newly inserted node
            
            new_g.add_edge(curr_num_node, to_insert_edge, 
                        # predicate=predicate,
                        txtemb = PredictDict[predicate],
                        distribute= distance, angle_AB = angle_AB,
                        angle_BA = angle_BA
                        )
    
    ## edit edge insertions
    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        while (True):
            curr_pair = random.sample(new_g.nodes(), 2)
            predicate=random.choice(global_edge_labels)
            # add edge to the newly inserted node
            # print("predicate: ",predicate)
            try:
                if ((curr_pair[0], curr_pair[1]) not in deleted_edges):
                    if ((curr_pair[0], curr_pair[1]) not in new_g.edges()):                    
                        new_g.add_edge(curr_pair[0], curr_pair[1], name=random.choice(global_edge_labels),
                                    txtemb = PredictDict[predicate],
                                    distribute= distance, 
                                    angle_AB = angle_AB,
                                    angle_BA = angle_BA                               
                        )
                        break
                else:
                    break
            except:
                print("EXCEPT")
    
    return target_ged, new_g


def PairDataset(file_path,train_num_per_row,  dataset, F0Dict,PredictDict, total_ged, train, args) : 
    # target_ged, new_g = graph_generation(Grph, F0Dict, global_edge_labels, total_ged)
    # subG, enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
    print("------- PairDataset ---------")

    g1_list = []
    g2_list = []
    ged_list = []

    length = len(dataset)
    cnt = 0
    if length != 0:
        print("tqdm - length: ", length)
        for i in tqdm(range(length)):            
            if train:
                print(" ---- mk GEVPair start ---- ")
                for _ in range(train_num_per_row): #일단 모델 사용해야해서 0, 1 나눔.. 
                    dataset[i].graph['gid'] = 0
                    # print(i, dataset[i])
                    if cnt > (train_num_per_row//2):
                        #F0Dict - global_node_list / Predict - global_edge_list -> 각 Dict의 key를 node와 edge의 후보군으로 사용
                        target_ged, new_g = graph_generation(dataset[i], F0Dict, PredictDict, total_ged)
                        # targetGED에 따라 생성한 그래프에 RPE 적용
                        # -> RPE 과정에서 concat될 때, concat 결과가 기존의 edge를 반영하는 것 같지 않음
                        # 해당 내용 확인 필요 
                        
                        new_g, enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
                        origin_g, enc_agg = mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 Feature 붙임
                        graph2 = new_g
                    else:
                        #text emb 값
                        target_ged, new_g = graph_generation(dataset[i], F0Dict, PredictDict, total_ged)
                        
                        new_g, new_enc_agg = mkNG2Subs(new_g, args, F0Dict, )  # Gs에 Feature 붙임
                        origin_g, origin_enc_agg = mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 Feature 붙임
                        graph2 = new_g

                    gev = [target_ged['nc'],target_ged['ec'],target_ged['in'],target_ged['ie'],]
                    graph2.graph['gid'] = 1

                    # print("origin_g: ", origin_g)
                    # print("new_g: ", graph2)
                    # print("gev: ", gev)

                    g1_list.append(origin_g)
                    g2_list.append(graph2)
                    ged_list.append(gev)  # gev로 바꿔서 넣음
                    cnt += 1
                cnt = 0
            else:
                r = random.randrange(length)
                dataset[r].graph['gid'] = 0
                target_ged, new_g = graph_generation(dataset[i], F0Dict, PredictDict, total_ged)

                new_g, new_enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
                origin_g, origin_enc_agg = mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 Feature 붙임

                graph2 = new_g
                g1_list.append(origin_g)
                g2_list.append(graph2)
                ged_list = [total_ged for _ in range(len(g2_list))]
                
            print("done - 1")

        else :
            print("length is 0 -> killed")

        with open("data/GEDPair/walk4_step3_ged10/walk{}_step{}_ged{}_{}.pkl".format(args.num_walks,args.num_steps,total_ged, file_path[-8:-4]), "wb") as fw:
            pickle.dump([g1_list, g2_list, ged_list], fw)
        # with open("data/GEDFeat/walk4_step3_ged10/walk{}_step{}_ged{}_{}.pkl".format(args.num_walks,args.num_steps,total_ged, file_path[-8:-4]), "wb") as fw:
        #     pickle.dump([subGFeatList, newGFeatList, ged_list], fw)
       
        # print("len(g1_list): ", g1_list)
        print("len(g1_list[0]): ", g1_list[0])

        g1_list = []
        g2_list = []
        ged_list = []
        
        # subGFeatList = []
        # newGFeatList = []


def process_file(file_path, F0Dict,PredictDict, total_ged, train, margs):
    train_num_per_row = 64      # Number of datasets created by one subgraph
    F0Dict = F0Dict.copy()
    PredictDict = PredictDict.copy()

    fpath = "data/scenegraph_1/"+str(file_path)    
    with open(fpath, 'rb') as file:
        data = pickle.load(file)
        content = data[0] #graphs
    
    PairDataset(file_path, train_num_per_row, *content, F0Dict,PredictDict, total_ged, train, margs)  # 여기에 특정 함수를 호출하고 결과를 저장
    
    return


def process_files_in_threads(file_list, num_threads, margs,):
    # global edge Label List 생성 -> Predicate dict
    with open('data/class_unique_textemb.pickle', 'rb') as f:  
       data  = pickle.load(f)
    F0Dict = data

    with open('data/predicate_unique_textemb.pickle', 'rb') as f:
        data  = pickle.load(f)
    PredictDict = data

    train = True
    total_ged = 10

    # 각 스레드를 생성하고 실행
    def process_files_thread(thread_files, F0Dict,PredictDict, total_ged, train,):
        for file in thread_files:
            process_file(file, F0Dict, PredictDict, total_ged, train, margs)

    # 각 스레드가 처리할 파일 개수를 계산
    files_per_thread = len(file_list) // num_threads
    # 스레드를 담을 리스트를 생성
    threads = []

    # 스레드 생성 및 실행
    for i in range(num_threads):
        start_idx = i * files_per_thread
        end_idx = start_idx + files_per_thread if i < num_threads - 1 else len(file_list)
        thread_files = file_list[start_idx:end_idx]

        # thread = threading.Thread(target=process_files_thread, args=(thread_files, F0Dict,PredictDict, total_ged, train,))
        thread = threading.Thread(target=process_files_thread, args=(thread_files, F0Dict,PredictDict, total_ged, train,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    margs = parser.parse_args()
    # main(margs)

    folderpath = "data/scenegraph_1"
    # folderpath = "data/scenegraph"
    file_list = os.listdir(folderpath)
    file_list = file_list[:10]
    # 5개의 스레드를 사용하여 파일 처리를 실행
    num_threads = 100
    process_files_in_threads(file_list, num_threads, margs)