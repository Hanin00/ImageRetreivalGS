import sys
import pickle
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import networkx as nx

from surel_gacc import run_walk
from utils.mkGraphRPE import *



'''
    S의 각 원소(워크셋)를 이용해 edge list를 생성하고, edge에 따라 node를 생성하는 방법으로 하나의 서브그래프로 병합
'''
# path 로 edge list 만들고 edge 추가하기; node path로 Graph 생성 
def mkMergeGraph(S, K, gT, nodeNameDict, F0dict):
    # --------- vvvvv 생성된 walk에 있는 모든 노드들을 각 노드의 id에 맞게 rpe 를 concat 후 mean pooling 한 값 ----------------------
    merged_K = np.concatenate([np.asarray(k) for k in K]).tolist()
    print(len)
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
# -------- ^^^^^ 생성된 walk에 있는 모든 노드들을 각 노드의 id에 맞게 rpe 를 concat 후 mean pooling 한 값 ----------------------

    # print("ke_ys : ",gT_mean.keys())
    # print("values: ",gT_mean.values())

   
# --------- vvvvv S의 각 워크를 edge list로 만들어 SubG에 더함 ----------------------
    subG = nx.Graph() # Graph 하나에 모두 합쳐야 해서 for 문 밖에 그래프 객체를 생성
    for idx, sPath in enumerate(S):
    # path to edgelist    == [[path[i], path[i+1]] for i in range(len(path)-1)]
        edgelist = list(zip(sPath, sPath[1:]))
        subG.add_edges_from(edgelist)
    print(sPath)
    print(S)
    print(subG.nodes())

    sys.exit()

# --------- vvvvv 생성된 각 subG에 F0 를 더함 + 각 노드의 rpe 값을 attribute로 추가함(GED를 계산하는 것이 아니라서 벡터로 있어도 됨)  ----------------------
# origini Id에 대한 처리. 해당 값을 제외하고 만들던가...
    for i in subG.nodes() :
        # subG.nodes[i].update(F0dict[nodeNameDict[i]]) #노드에 해당하는 
        subG.nodes[i]['f0'] = F0dict[nodeNameDict[i]]
        subG.nodes[i]['rpe'] = gT_mean[i]

    print(subG.nodes(data=True))
    print(subG)
    return subG


#node 각각에 feature 추가할 때 structural feature 로 enc 값 추가해야함
# mk RPE encoding, subgraphs
'''
    새로운 그래프 new_에 대해 run_walk를 이용해 rpe enc값을 추출하고, 특징값을 concat + 하나의 서브 그래프로 병합
     -> 해당 subgraph에 대한 rpe embedding값을 얻을 수 있고, 새로 생성해서 사라진 name과 node idx 값 외의 속성을 추가(F0, rpe값)
'''
# def mkNG2Subs(G, args, F0dict, originGDict):
def mkNG2Subs(G, args, F0dict):
    # originGraph의 feature를 가져옴
    nmDict = dict((int(x), y['name'] ) for x, y in G.nodes(data=True)) # id : name 값인 Node Name Dict
    subGList, subGFeatList = [], []

    Gnode = list(G.nodes())
    print("Gnode : ",Gnode)

    G_full = nx2csr(G)
    ptr = G_full.indptr
    neighs = G_full.indices
    num_pos, num_seed, num_cand = len(set(neighs)), 100, 5
    candidates = G_full.getnnz(axis=1).argsort()[-num_seed:][::-1]
    print("candidates: ",candidates)
    rw_dict = {}
    B_queues  = []

    # for r in range(1, args.repeat + 1): # 모든 노드에 대해 한 번씩 할거라 repeat 필요 X
    batchIdx, patience = 0, 0
    pools = np.copy(candidates)
    print("candidates : ",candidates)

    np.random.shuffle(B_queues)
    # while True:
    # if r <= 1:
    B_queues.append(sorted(run_sample(ptr,  neighs, Gnode, thld=1500))) # pool를 인자로 넣어 모든 노드에 대해 수행하도록 함
    print("B_queues : ",B_queues)

    B_pos = B_queues[batchIdx]
    print("B_pos :", B_pos)

    B_w = [b for b in B_pos if b not in rw_dict]
    if len(B_w) > 0:
        walk_set, freqs = run_walk(ptr, neighs, B_w, num_walks=args.num_walks, num_steps=args.num_steps - 1, replacement=True)
    node_id, node_freq = freqs[:, 0], freqs[:, 1]
    rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))
    batchIdx += 1

    # obtain set of walks, node id and DE (counts) from the dictionary
    S, K, F = zip(*itemgetter(*B_pos)(rw_dict))

    print(rw_dict.values())
    


    # print("S: ",S[0]): 0번 노드로 시작하는 서브 그래프( walks set ) -> len(walks set): num_walks*(num_step+1)
    # print("K: ",K[0]): 0번 노드로 시작하는 서브 그래프의 .nodes() <- 이때 해당 노드의 F0 값은 없음
    # print("F: ",F[0]): S[0]의 Feature 값; 

    F = np.concatenate(F) #([[[0] * F.shape[-1]], F])   # rpe encoding 값(step 들만)
    mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F]))  #.to(device) # walk를 각 노드에 맞춰서 concat
    gT = mkGutils.normalization(mF, args)

    listA = [a.flatten().tolist() for a in K] 
    flatten_listA = list(itertools.chain(*listA))  # 35*12

    subG = mkMergeGraph (S, K, gT, nmDict, F0dict)

    print(gT) #이 값 concat
    gT_concatenated = torch.cat((gT, gT), axis=1)
    enc_agg = torch.mean(gT_concatenated, dim=0)
    print("enc_agg: ", enc_agg)


    #   print(G.nodes())
    #   print(listA)
    #   print("S: ",S) 
    #   print("K: ",K)
    #   print("gT: ",gT)

    #   print("S: ",len(S) )
    #   print("K: ",len(K))
    #   print("mF: ",len(mF))
    #   print("gT: ",len(gT))

    sys.exit()




    subG, subGF = mkSubGraph(S, K, gT, nodeDict)


    subGList+=subG
    subGFeatList+=subGF

    '''
    listA = [a.flatten().tolist() for a in K] 
    flatten_listA = list(itertools.chain(*listA))  # 35*12
    print(len(flatten_listA))
    print("K의 노드 개수: ", len(flatten_listA))
        -> K에는 중복되는 노드 id가 있지만, 각 노드는 서브그래프(S[i]) 내에서 다른 RPE값을 가짐(F[i], mF[i])

        때문에 이를 토대로 서브 그래프 생성(mkPathGraph) 후 해당 RPE 값을  attribute로 부여해줘야하고, 
        또 맨 위에서 만들어 둔 nodeDict를 이용해서 name에 따른 feature 값인 F0값을 부여해줘야함      
        '''

    return subGList, subGFeatList






# def return_eq(node1, node2):
#     return node1['type']==node2['type']
'''
    targer GEV 를 랜덤으로 생성하고 그에 맞는 pair graph를 생성
'''
def graph_generation(graph, F0Dict, global_edge_labels, total_ged=0):
    new_g = deepcopy(graph)
    
    global_labels = list(F0Dict.keys())

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

    print('target ged :', target_ged)

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
            if (toassigned_new_edgetype != new_g.edges()[idx]['name']):
                break
        new_g.edges()[idx]['name'] = toassigned_new_edgetype

    ## edit node insertions
    for num in range(target_ged['in']):
        curr_num_node = new_g.number_of_nodes()
        to_insert_edge = random.sample(new_g.nodes(), 1)[0]
        new_g.add_node(str(curr_num_node), label=str(curr_num_node), name=random.choice(global_labels))
        # add edge to the newly inserted ndoe
        new_g.add_edge(str(curr_num_node), to_insert_edge, name=random.choice(global_edge_labels))

    ## edit edge insertions
    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        while (True):
            curr_pair = random.sample(new_g.nodes(), 2)
            if ((curr_pair[0], curr_pair[1]) not in deleted_edges):
                # print('poten edge', curr_pair[0], curr_pair[1])
                if ((curr_pair[0], curr_pair[1]) not in new_g.edges()):
                    # print('added adge', curr_pair[0], curr_pair[1])
                    new_g.add_edge(curr_pair[0], curr_pair[1], name=random.choice(global_edge_labels))
                    break

    # print('Total target ged', target_ged['nc'] + target_ged['ec'] + target_ged['in'] + target_ged['ie'], total_ged)
    # print('----------------------------------------------------------------------------')

    return target_ged, new_g



def load_generated_graphs(dataset_name, file_name='generated_graph_500'):
    dir = 'dataset/' + dataset_name + '/' + file_name
    g = open(dir, 'rb')
    generated_graphs = pickle.load(g)
    g.close()
    return generated_graphs

'''
    originGDict : 대상 Graph의 node의 name - attribute 값(왜) ; 이름을 가지고 node relabeling을 하니까.  ; origin Id 가 좀 걸리는데..  
    F0Dict : global node name - F0 embedding
'''
def PairDataset(Grph, F0Dict,global_edge_labels, total_ged) : 
    
    # originGDict = dict((x, y ) for x, y in Grph.nodes(data=True))
    ## Here is an example of generating graph pair (graphs[0], new_g) and their corresponding target_ged;
    ## Note that the input "total_ged" here is a randomly sampled value, it is only for reference and may not exactly equal to the returned final target_ged
    target_ged, new_g = graph_generation(Grph, F0Dict, global_edge_labels, total_ged)
                                        #  total_ged=random.randint(18,18))
    #  Origin 1 Graph 기준으로 node id - F0, origin Id 등의 origin 
    graph1 = []
    graph2 = []
    ged = []
    graph1.append(Grph)
    graph2.append(new_g)
    ged.append(target_ged)
   
    print(Grph.nodes(data=True))
    print(target_ged)
    print("1:", new_g.nodes(data=True))
    

    #새로운 graph 생성.
    graph_pair = []
    
    # subGList, subGFeatList = mkNG2Subs(new_g, args, F0Dict, originGDict)
    
    print("origins_g: ", Grph.nodes())
    print("new_g : ", new_g.nodes())

    subGList, subGFeatList = mkNG2Subs(new_g, args, F0Dict)
    print("subGList: ", subGList) 
    print("subGFeatList: ", subGFeatList)
    
    print("1:", new_g.nodes(data=True))
    
    
    




     


#    new_g에 대해 rpe encoding; 
    # csrNG = nx2csr(new_g)
    # ptr = csrNG.indptr
    # neighs = csrNG.indices
    # candidates = ptr.getnnz(axis=1).argsort()[-100:][::-1]
    # pools = np.copy(candidates)


    # rw_dict = {}
    # B_queues  = []
    # batchIdx, patience = 0, 0

    # B_w = list(new_g.nodes())
    # B_pos = B_queues[batchIdx]
    # B_queues.append(sorted(run_sample(ptr,  neighs, pools, thld=1500))) # pool를 인자로 넣어 모든 노드에 대해 수행하도록 함
    # B_pos = B_queues[batchIdx]
    # B_w = [b for b in B_pos if b not in rw_dict]

    
    # walk_set, freqs = run_walk(ptr, neighs, B_w, num_walks=args.num_walks, num_steps=args.num_steps - 1, replacement=True)
    # node_id, node_freq = freqs[:, 0], freqs[:, 1]
    # rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))
    # S, K, F = zip(*itemgetter(*B_w)(rw_dict))




    # print("walk_set: ",walk_set)
    # print("freqs: ",freqs)





    # 이렇게만 해서 freq에 따라 concat하면 되는 거 아닌가..?
    # pools = 5
    # seeds = np.random.choice(pools, 5, replace=False)
    # start = time.time()
    # # totalData = []
    # metaData = [] # 각 originGId 당 생성된 subGList의 개수가 들어감 - originGId, len(subGList)
    # totalSubG = []
    # totalSubGFeat = []

    # for originGId, G in enumerate(tqdm(new_g)): # 원본 데이터의 오류로 없는 그래프가 종종 있음.try catch 해서 넘기기
    #     if(len(G.nodes()) != 0):
    #         subGs, subGFs = mkSubs(G, args, seeds)








def main(args):
# #node type의 class들 -> name, feature를 전에 만들어놓은 dict를 이용해서 넣을 것; 동일 그래프 내의 node로만 생성하거나, 전체 node에 대해 생성(우선)

    # subgraph  load에 맞춰서 변경해야함
    # with open('dataset/img100_walk4_step3/subG.pkl', 'rb') as f:  # 
    #     graphs = pickle.load(f)
    # with open('dataset/img100_walk4_step3/subG_100.pkl', 'wb') as f:
    #     pickle.dump(graphs[:100], f)   

    # with open('dataset/img100_walk4_step3/subGFeat.pkl', 'rb') as f:  # 
    #     feats = pickle.load(f)
    # with open('dataset/img100_walk4_step3/subGFeat_100.pkl', 'wb') as f:
    #      pickle.dump(feats[:100], f)   

    # sys.exit()
    with open('dataset/img100_walk4_step3/subG_100.pkl', 'rb') as f:  # 
        graphs = pickle.load(f)
    with open('dataset/img100_walk4_step3/subGFeat_100.pkl', 'rb') as f:  # 
        feats = pickle.load(f)


    #subgraph  load에 맞춰서 변경해야함
    with open('dataset/totalEmbDictV3_x100.pickle', 'rb') as f:  
       embDict  = pickle.load(f)

    # global_node_labels = list(embDict.keys())
    global_edge_labels = [0, 0]

    Grph  = graphs[50]
    total_ged=random.randint(1,1)
    
    #Graph 는 이미 RPE 값을 가지고 있음; 해당 RPE 값의 Feature도 동일.. 
    #새로 만든 그래프의 RPE를 구하고, Origin Graph의 RPE를 구한 것처럼 concat, mean pooling으로
    # 해당 subgraph의 feature 값을 구해야함
    # 의문점; 이러면 node의 structural Feature만 있고, node 각각의 feature는 embedding이 안된 것 아닌가?
    # node adj 등을 이용해 얻어낸 기존 방법에서 pred emb할 때, concat을 해줘야 하나?
    PairDataset(Grph, embDict,global_edge_labels, total_ged)




    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    main(args)


    



