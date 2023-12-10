from astar_ged.src.distance import ged, normalized_ged

import multiprocessing as mp
import pickle
import random

import sys
import numpy as np
import time


def make_pkl(dataset, queue, train_num_per_row, max_row_per_worker, train):
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

                    d = ged(dataset[i], graph2, 'astar',
                            debug=False, timeit=False)
                    d = normalized_ged(d, dataset[i], graph2)
                    
                    g1_list.append(dataset[i])
                    g2_list.append(graph2)
                    ged_list.append(d)
                    cnt += 1
                cnt = 0
            else:
                dataset[i].graph['gid'] = 0
                r = random.randrange(length)
                dataset[r].graph['gid'] = 1
                d = ged(dataset[i], dataset[r], 'astar',
                        debug=False, timeit=False)
                d = normalized_ged(d, dataset[i], dataset[r])
                g1_list.append(dataset[i])
                g2_list.append(dataset[r])
                ged_list.append(d)

        with open("dataset/GEDPair/rpe_splited_v3_x1000_walk4_step3/{}_{}.pkl".format(s, e), "wb") as fw:
            pickle.dump([g1_list, g2_list, ged_list], fw)
        g1_list = []
        g2_list = []
        ged_list = []


def main(train):
    mp.set_start_method('spawn')
    q = mp.Queue()
    train_num_per_row = 64      # Number of datasets created by one subgraph
    max_row_per_worker = 64     # Number of Subgraphs processed by one processor
    number_of_worker = 80       # Number of processor

#    with open("data_rp/rpe_v3_x1000_SubG.pkl", "rb") as fr:
    with open("dataset/rpe_splited_subgraph/rpe_v3_x1000_walk4_step2_SubG_129897.pkl", "rb") as fr:
        dataset = pickle.load(fr)

    #print(dataset[0].nodes[2])
    #{'r0': 1, 'r1': 0, 'r2': 0, 'r3': 0, 'r4': 0, 'name': 'gym_shoe', 'originId': 5048, 'f0': -0.19570904970169067}

    total = dataset

    start = time.time()
    #전체 129897 을 64로 나눠서 4/5지점부터 GED Pair 생성
    for i in range(77938, len(total), max_row_per_worker):
    #for i in range(len(total), 0, -max_row_per_worker):  #mk reverse
        q.put(i)

    workers = []
    for i in range(number_of_worker):
        worker = mp.Process(target=make_pkl, args=(
            total, q, train_num_per_row, max_row_per_worker, train))
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    end = time.time()


    print("time: ", end-start)


if __name__ == "__main__":
    main(True)
    # python3 -m common.data_batch
