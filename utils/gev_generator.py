import sys
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



def normalize_labels(labels):
    min_vals = torch.min(labels, dim=0)[0]
    max_vals = torch.max(labels, dim=0)[0]
    normalized_labels = (labels - min_vals) / (max_vals - min_vals)
    return normalized_labels

'''
    S의 각 원소(워크셋)를 이용해 edge list를 생성하고, edge에 따라 node를 생성하는 방법으로 하나의 서브그래프로 병합
'''
# path 로 edge list 만들고 edge 추가하기; node path로 Graph 생성 
def mkMergeGraph(S, K, gT, nodeNameDict, F0dict, nodeIDDict):
    print("-------mkMergeGraph-------")
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
#그래프를 여기서 만드는 게 아니라 각 노드별 rpe 값만 반환

# # -------- ^^^^^ 생성된 walk에 있는 모든 노드들을 각 노드의 id에 맞게 rpe 를 concat 후 mean pooling 한 값 ----------------------
#     # print("ke_ys : ",gT_mean.keys())
#     # print("values: ",gT_mean.values())

# # --------- vvvvv S의 각 워크를 edge list로 만들어 SubG에 더함 ----------------------
#     subG = nx.Graph() # Graph 하나에 모두 합쳐야 해서 for 문 밖에 그래프 객체를 생성
#     for idx, sPath in enumerate(S):
#     # path to edgelist    == [[path[i], path[i+1]] for i in range(len(path)-1)]
#         sPath = [nodeIDDict[i] for i in sPath]
#         edgelist = list(zip(sPath, sPath[1:]))
#         subG.add_edges_from(edgelist)
    
#     # print(subG.nodes(data=True))

# # --------- vvvvv 생성된 각 subG에 F0 를 더함 + 각 노드의 rpe 값을 attribute로 추가함(GED를 계산하는 것이 아니라서 벡터로 있어도 됨)  ----------------------
# # todo origini Id에 대한 처리. 해당 값을 제외하고 만들던가...
#     for i in subG.nodes() :
#         # subG.nodes[i].update(F0dict[nodeNameDict[i]]) #노드에 해당하는 
#         subG.nodes[i]['f0'] = F0dict[nodeNameDict[int(i)]]
#         subG.nodes[i]['name'] = nodeNameDict[int(i)]
#         subG.nodes[i]['rpe'] = gT_mean[i]

#     # print(subG.nodes(data=True))
#     # print(subG)
#     return subG


#node 각각에 feature 추가할 때 structural feature 로 enc 값 추가해야함
# mk RPE encoding, subgraphs
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
    print("G_full: ", G_full) #각 노드간 연결을 표현할 뿐, 그 외는 표현 불가능

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





# def return_eq(node1, node2):
#     return node1['type']==node2['type']
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
            try:
                if (toassigned_new_edgetype != new_g.edges()[idx]['name']):
                    break
            except:
                continue
                # print('line 210 -  edge name')
        new_g.edges()[idx]['name'] = toassigned_new_edgetype

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
        print("edge feature cal")

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

            # add edge to the newly inserted ndoe
            new_g.add_edge(curr_num_node, to_insert_edge, 
                        predicate=random.choice(global_edge_labels),
                        distribute= distance, angle_AB = angle_AB,
                        angle_BA = angle_BA 
                        )
        
    ## edit edge insertions
    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        while (True):
            curr_pair = random.sample(new_g.nodes(), 2)
            if ((curr_pair[0], curr_pair[1]) not in deleted_edges):
                if ((curr_pair[0], curr_pair[1]) not in new_g.edges()):
                    new_g.add_edge(curr_pair[0], curr_pair[1], name=random.choice(global_edge_labels))
                    break
    return target_ged, new_g


def load_generated_graphs(dataset_name, file_name='generated_graph_500'):
    dir = 'dataset/' + dataset_name + '/' + file_name
    g = open(dir, 'rb')
    generated_graphs = pickle.load(g)
    g.close()
    return generated_graphs

'''
    originGDict : 대상 Graph의 node의 name - attribute 값(왜) ; 이름을 가지고 node relabeling을 하니까.  ; todo origin Id 가 좀 걸리는데..  
    F0Dict : global node name - F0 embedding
'''
# def PairDataset(queue, train_num_per_row, max_row_per_worker, dataset,feats, F0Dict,PredictDict, total_ged, train,args) : 
def PairDataset(queue, train_num_per_row, max_row_per_worker, dataset, F0Dict,PredictDict, total_ged, train,args) : 
    # target_ged, new_g = graph_generation(Grph, F0Dict, global_edge_labels, total_ged)
    # subG, enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
    g1_list = []
    g2_list = []
    ged_list = []

    subGFeatList = []
    newGFeatList = []

    cnt = 0
    length = len(dataset)
    while True:
        if queue.empty():
            break
        num = queue.get()
        if length-num > max_row_per_worker:
            s = num
            e = num + max_row_per_worker
        else:
            s = num
            e = len(dataset)
        for i in range(s, e):
            if train:
                for _ in range(train_num_per_row): #일단 모델 사용해야해서 0, 1 나눔.. 
                    dataset[i].graph['gid'] = 0
                    # print(i, dataset[i])
                    if cnt > (train_num_per_row//2):
                        #F0Dict - global_node_list / Predict - global_edge_list -> 각 Dict의 key를 node와 edge의 후보군으로 사용
                        target_ged, new_g = graph_generation(dataset[i], F0Dict, PredictDict, total_ged)

                        # print("origin_g: ", dataset[i])
                        # print("new_g: ", new_g)
                        # print("target_ged: ", target_ged)
                        # print("total_ged: ", total_ged)

                        # targetGED에 따라 생성한 그래프에 RPE 적용
                        # -> RPE 과정에서 concat될 때, concat 결과가 기존의 edge를 반영하는 것 같지 않음
                        # 해당 내용 확인 필요 
                        
                        new_g, enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
                        origin_g, enc_agg = mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 Feature 붙임

                        sys.exit()

                        graph2 = subG
                    else:
                        #text emb 값
                        target_ged, new_g = graph_generation(dataset[i], F0Dict, PredictDict, total_ged)
                    
                        # print("여기")
                        # print("origin_g: ", dataset[i])
                        # print("new_g: ", new_g)
                        # print("target_ged: ", target_ged)
                        # print("total_ged: ", total_ged)

                        # print("origin_g.nodes(): ", dataset[i].nodes(data=True))
                        # print("new_g.nodes(): ", new_g.nodes(data=True))
                        # print("origin_g.edges(): ", dataset[i].edges(data=True))
                        # print("new_g.edges(): ", new_g.edges(data=True))

                        new_g, new_enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
                        origin_g, origin_enc_agg = mkNG2Subs(dataset[i], args, F0Dict)  # Gs에 Feature 붙임

                        # print("new_g ")
                        # print("nodes: ", new_g.nodes(data=True))
                        # print("edges: ", new_g.edges(data=True))

                        # print("origin_g: ", origin_g)
                        # print("nodes: ", origin_g.nodes(data=True))
                        # print("edges: ", origin_g.edges(data=True))

                        graph2 = new_g

                    gev = [target_ged['nc'],target_ged['ec'],target_ged['in'],target_ged['ie'],]
                    # labels = [[label[key] for key in ['nc', 'ec', 'in', 'ie']] for label in target_ged]
                    # print("*-*-*-*")
                    # print("gev : ", gev)
                    # sys.exit()
                    # print("target_ged : ", target_ged)
                    # print("target_ged : ", target_ged)
                    # print("*-*-*-*")

                    graph2.graph['gid'] = 1
                    # d = ged(dataset[i], graph2, 'astar',
                    #         debug=False, timeit=False)
                    g1_list.append(dataset[i])
                    g2_list.append(graph2)
                    ged_list.append(gev)  # gev로 바꿔서 넣음

                    # subGFeatList.append(feats[i])
                    # newGFeatList.append(enc_agg)

                    cnt += 1
                cnt = 0
            else:
                r = random.randrange(length)
                dataset[r].graph['gid'] = 0
                target_ged, new_g = graph_generation(dataset[r], F0Dict, PredictDict, total_ged)
                subG, enc_agg = mkNG2Subs(new_g, args, F0Dict)  # Gs에 Feature 붙임
                graph2 = subG
                g1_list.append(dataset[r])
                g2_list.append(subG)
                ged_list.append(target_ged)

                # subGFeatList.append(feats[r])
                # newGFeatList.append(enc_agg)

            # 정규화 여기서
            max_value = 9.0
            ged_tensor = torch.tensor(ged_list)
            ged_norm_list = ged_tensor / max_value
            # print("ged_norm_list: ",ged_norm_list)

            # sys.exit()
            # print("ged_norm_list: ",ged_norm_list)


            with open("dataset/GEDPair/img100_walk4_step3_0512_ged6/__{}_{}.pkl".format(s, e), "wb") as fw:
                pickle.dump([g1_list, g2_list, ged_norm_list], fw)
            with open("dataset/GEDFeat/img100_walk4_step3_0512_ged6/__{}_{}.pkl".format(s, e), "wb") as fw:
                pickle.dump([subGFeatList, newGFeatList, ged_norm_list], fw)
        
            g1_list = []
            g2_list = []
            ged_list = []
            
            subGFeatList = []
            newGFeatList = []




'''
    mkGraphRPE.py에서 SceneGraph를 Random walk base로 나누고, RPE를 계산한다. 
    walk를 인접한 노드끼리 합쳐 subgraph를 생성한다. (워크가 크면 노드 수가 증가할 가능성이 높음. 인접한 walk가 많을 수 O)
    RPE 값을 concat하고 mean pooling해 해당 subgraph의 structural feature를 embedding가능
    
    각 노드에 subgraph 에서 해당 노드가 갖는 rpe 값들을 concat하고 mean pooling해 subgraph내 각 노드의 structural feature를 Attribute 값으로 할당했음

    이렇게 만들어진 subgraph에 target_gev와 대응하는 subgraph를 생성함
    
    PairDataset -> graph_generation -> mkNG2Subs -> mkMergeGraph
    1. PairDataset에서 그래프를 생성하고 이를 데이터셋 형태에 맞게 저장함 (기존 데이터셋의 경우 batch가 64였음..)
    2. graph_generation에서 target_gev를 random 하게 생성하는데, 순서에 따라 개수가 조절되므로 고립되는 노드가 없게했다고 설명을 들었음
    3. mkNG2Subs에서는 만들어진 서브 그래프의 rpe를 구하고 
    4. rpe는 구했는데, mkGraphRPE에서 만든 하나의 서브그래프와 대응되는 서브그래프'를 randomwalk한 것이므로 각 값을 할당해줌. 
    -> 맨 위의 rpe 함수를 붙이기 전에 주석을 달고 commit....
    ## GEV를 나눠서 학습할 때랑 GED로 만들어서 학습할 때 뭐가 더 잘 찾는지 궁금함
    ### GEV를 뭐에 대해 normalize..?

'''


def main(margs):
    # global edge Label List 생성 -> Predicate dict

    with open('data/Vidor/scenegraph/0_2754378442_6188920051.pkl', 'rb') as f:   # time:  74.21744275093079
        data = pickle.load(f)
    with open('data/Vidor/class_unique_textemb.pickle', 'rb') as f:  
       embDict  = pickle.load(f)
    with open('data/Vidor/predicate_unique_textemb.pickle', 'rb') as f:
        predDict = pickle.load(f)

  # for idx, file in enumerate(data[0]): #graph List List
  #    print("file: ", file)
  #    if len(file)!= 0:      
  #       print(data[0][idx]) #graph  
  #       print(data[1][idx]) #json file name
  #       print(data[2][idx]) #fid

    graphs = []
    cnt = 0
    for idx, file in enumerate(data[0]): #graph List List
        #  print("file: ", file)
        if len(file)!= 0:    
            graphs.extend(data[0][idx])
    
    #일단 당장 할 거..!
    graphs = graphs
    # feats = feats
    # PairDataset(Grph, embDict,global_edge_labels, total_ged)
    print("--- data_load ---")

    mp.set_start_method('spawn')
    q = mp.Queue()
    train_num_per_row = 64      # Number of datasets created by one subgraph
    max_row_per_worker = 64     # Number of Subgraphs processed by one processor
    # number_of_worker = 40       # Number of processord
    number_of_worker = 40       # Number of processord
    total = graphs
    # global_node_labels = list(embDict.keys())
    # global_edge_labels = list(predDict.keys())

    total_ged=random.randint(18, 18)
    train = True

    print("start")
    start = time.time()
    for i in range(0, len(total), max_row_per_worker):
        q.put(i)

    print("queue")

    workers = []
    for i in range(number_of_worker):
        worker = mp.Process(target=PairDataset, args=(
            q, train_num_per_row, max_row_per_worker, graphs, embDict, predDict, total_ged, train, margs))
            # q, train_num_per_row, max_row_per_worker, graphs,feats,  embDict, predDict, total_ged, train, margs))
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    end = time.time()
    print("time: ", end-start)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    margs = parser.parse_args()

    main(margs)

