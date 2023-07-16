
from astar_ged.src.distance import ged, normalized_ged

import multiprocessing as mp
import pickle
import random

import sys
import numpy as np
import time

'''
    astar 계산 시 rpe 값도 특징으로 보도록 변경

    동일한 subgraph로 비교할 수 있도록, (random walk시 변경될 수 있으니까)
    r0,... r4 만들어둔 것 말고 
    rpe_v3_x1000_step1_SubG 에서 node attribute만 변경함

'''


def make_pkl(dataset, queue, train_num_per_row, max_row_per_worker, train, filePath, bigGED):
    '''Make pickle file(create train dataset)
    Format of one train data is graph1, graph2, ged of graph1 and graph2.
    This process is to create a train dataset from subgraphs.
    It creates `max_row_per_worker` train dataset per subgraph.
    The train dataset have positive and negative dataset at the same rate.
    The positive dataset is created by generating a new subgraph that removes
    particular node from the original subgraph.
    The negative dataset is created by using other subgraphs.
    And then, it stores the processed train dataset per worker in pickle file

    dataset: Subgraphs generated from scene graph
    queue: Counter for checking processed subgraphs (rule flag)
    train_num_per_row: Number of datasets created by one subgraph
    max_row_per_worker: Number of subgraphs processed by one processor
    train: Whether it's learning or not (True or False)

    return: list of subgraph1, list of subgraph2, list of ged
    '''
    g1_list = []
    g2_list = []
    ged_list = []
    ged_unnorm_list = []
    cnt = 0
    length = len(dataset)

    '''
    GED가 클 경우4
     
    1. node 수가 차이가 많이 나는 경우
    2. edge 수가 차이 나는 경우
    3. node name이 일치하지 않는 경우

    GED가 작을 경우
    1. node와 edge수가 동일한 경우     
    2. node 가 동일할 경우
    
    '''

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
                for _ in range(train_num_per_row):
                    dataset[i].graph['gid'] = 0
                    # print(i, dataset[i])
                    if cnt > (train_num_per_row//2):
                        # print(a, b)
                        l = list(dataset[i].nodes())
                        l.remove(random.choice(l))
                        graph2 = dataset[i].subgraph(l)
                        # print(1, r)
                    else:
                        r = random.randrange(length)
                        graph2 = dataset[r]

                    graph2.graph['gid'] = 1
                    
                    
                    # 조건 여기서 걸어야 함
                    g1nlist = dataset[i].nodes(data = 'name')
                    g2nlist = graph2.nodes(data = 'name')                    

                    g1elist = dataset[i].edges()
                    g2elist = graph2.edges()               


#---------------------------------------  조건 ---------------------------------------
                    if bigGED : #node diff
                        if(len(g1nlist-g2nlist) == len(g1nlist)): # 노드가 아예 다른 것 # GED 큰 것
                            d = ged(dataset[i], graph2, 'astar',
                            debug=False, timeit=False)
                            print("unnorm d: ",d)
                            ged_unnorm_list.append(d)
                            d = normalized_ged(d, dataset[i], graph2)

                            g1_list.append(dataset[i])
                            g2_list.append(graph2)
                            ged_list.append(d)
                            cnt += 1
                    else: #node same
                        if(len(g1nlist-g2nlist) <= 1) &  (len(g1elist - g2elist) < len(g1elist)-1 ): # 노드가 하나 이하로 다르거나 같고 edge가 하나 이상 다른 것 # GED 가 작은 것
                            d = ged(dataset[i], graph2, 'astar',
                            debug=False, timeit=False)
                            print("unnorm d: ",d)
                            ged_unnorm_list.append(d)
                            d = normalized_ged(d, dataset[i], graph2)

                            g1_list.append(dataset[i])
                            g2_list.append(graph2)
                            ged_list.append(d)
                            cnt += 1
#---------------------------------------  data append ---------------------------------------
                        
                    cnt = 0            
                    
            else:
                print("else")
                dataset[i].graph['gid'] = 0
                r = random.randrange(length)
                dataset[r].graph['gid'] = 1
                d = ged(dataset[i], dataset[r], 'astar',
                        debug=False, timeit=False)
                ged_unnorm_list.append(d)
                d = normalized_ged(d, dataset[i], dataset[r])
                g1_list.append(dataset[i])
                g2_list.append(dataset[r])
                ged_list.append(d)


            if (len(g1_list) >=1 & len(g2_list )>=1 ):
                print("g1_list: ", g1_list)
                print("g2_list: ", g2_list)
                print("d: ", ged_unnorm_list )
                with open(filePath + "_unnorm/{}_{}.pickle".format(s, e), "wb") as fw:      
                    pickle.dump([g1_list, g2_list, ged_unnorm_list], fw)
                with open( filePath + "/{}_{}.pickle".format(s, e), "wb") as fw:
                    pickle.dump([g1_list, g2_list, ged_list], fw)
        g1_list = []
        g2_list = []
        ged_list = []
        ged_unnorm_list = []


def chngAttr(Grp):
    # vector로 된 rpe를 개별적인 attr로 변경
    for idx, nodes in enumerate(Grp.nodes): # 그래프 내 노드 
        #print("nodes: ", nodes)
        rpe = Grp.nodes[nodes]['rpe']
        #print("rpe: ", rpe)
        attr_list = [f"r{ridx}" for ridx in range(4 +1)] # 4: num_step
        for ridx, fname in enumerate (attr_list):   
         #   print(Grp.nodes[nodes])
          #  print("fname: ", fname)
           # print("rpe[ridx]: ", rpe[ridx])
            Grp.nodes[nodes][fname] = int(rpe[ridx])
        
    return Grp


def main(train):
    '''
        withR0 = 0 # 1이면 RPE -> r0, ...,r4
        bigGED = 0 # 0: node same / 1: node diff
    
        WITH R0가 0인경우 distance.py - get_result의 ln을 23로 변경
        0,0 : f0, big(diff)
        0,1 : f0, small(same)

        WITH R0가 1인경우 distance.py - get_result의 ln을 27로 변경
        1,0 : r1, big(diff)
        1,1 : r0, small(same)

    '''


    withR0 = 0 # 
    bigGED = 0 #
    #node same, small ged 
    

    mp.set_start_method('spawn')
    q = mp.Queue()
    train_num_per_row = 64      # Number of datasets created by one subgraph
    max_row_per_worker = 64     # Number of Subgraphs processed by one processor
    number_of_worker = 80       # Number of processor

    with open("data/rpe_vector_x1000_step1_SubG.pkl", "rb") as fr: # vector
        dataset = pickle.load(fr)
    if(bigGED) : 
        print("this")
        dataset = dataset[:40] # 차이가 많이 나는 경우가 많아서 겁나 오래걸림
    else:
        dataset = dataset[:100] #
    
    # ------------------------- withR0 = 0 # 1이면 RPE -> r0, ...,r4 -------------------------
    if (withR0):
        dataset = [chngAttr(grp) for grp in dataset]

    total = dataset
    start = time.time()
    for i in range(0, len(total), max_row_per_worker):
        q.put(i)

    if (withR0): # 
        filePath = "data/ged_f0_rpe/02ged_f0_rpe_node"    
    else: 
        filePath = "data/ged_f0/02ged_f0_node"   

    # ------------------------- bigGED = 0 # 0: node same / 1: node diff-------------------------
    if bigGED :  
        filePath = filePath+'_diff'
    else:
        filePath = filePath + '_same_edge_1diff'
    
    print("filePath: ", filePath)

    make_pkl(total, q, train_num_per_row, max_row_per_worker, train, filePath, bigGED)




#  for i in range(number_of_worker):
#      worker = mp.Process(target=make_pkl, args=(
#          total, q, train_num_per_row, max_row_per_worker, train))
#      workers.append(worker)
#      worker.start()

#    for worker in workers:
#        worker.join()
    end = time.time()
    print("time: ", end-start)


import os
import glob
import pickle

def merge_pickles(mergeFilePath):
    pickle_dir = mergeFilePath
    merged_file = mergeFilePath + ".pkl"

    data = []
    for file in glob.glob(os.path.join(pickle_dir, "*.pickle")):
        with open(file, 'rb') as f:
            data += pickle.load(f)
    with open(merged_file, 'wb') as f:
        pickle.dump(data, f)

def datasetChk(filePath):
    with open( filePath + ".pkl", "rb") as fr:
        dataset = pickle.load(fr)
    print(len(dataset))

    #merge 한 경우 - ged_f0_node_same_edge_1diff / ged_f0_node_same_edge_1diff_unnorm
    for i in range(0, 20*3, 3):
        print(dataset[i][0].nodes(data=True))
        print(dataset[i+1][0].nodes(data=True))
        print(dataset[i+2][0])


    #merge 얀 한 경우 - ged_f0_node_diff / ged_f0_node_diff_unnorm
    #for i in range(20):
    #    print(dataset[0][i].nodes(data=True))
    #    print(dataset[1][i].nodes(data=True))
    #    print(dataset[2][i])

    #merge 한 경우 - ged_f0_node_same_edge_1diff_unnorm / ged_f0_node_same_edge_1diff

    
if __name__ == "__main__":
    
    main(True)

    #merge_pickles(filePath)
    #datasetChk()

    # python3 -m common.data_batch