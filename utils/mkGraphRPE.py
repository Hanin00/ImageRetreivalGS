'''
  SUREL에 나온 RPE encoding을 이용해 subraph의 feature를 추출; 
  각 워크는 노드 4개를 가지며, 모든 노드로부터 출발하도록 변경할 필요..?< 왜 모든 노드에 대해 한 게 아니지..?

    freqs[:, 0]: walkset 에 나오는 node id들의 집합
    freqs[:, 1]:  walkset 에 나오는 node id 들의 encoding된 값; 동일한 노드여도 다른 값을 가질 수 있음; 워크마다 다른 구조적 특징값을 가지니까
    => 워크 셋이 아닌 워크로 서브 그래프를 만들고, 해당 서브그래프에 freqs[:, 1]의 enc 값을 특징으로 부여함; 이렇게 해서 subgraph split을 하고,
    이에 대한 ged 계산은 CBIR-SG에서 사용한 astar GED 코드를 사용

0429 1000개 그래프에 대해 서브그래프와 GED 셋 만들고, 
  서브 그래프 자체의 feature에 structural Graph feature를 추가해서 학습 가능하도록 코드 변경하기..

  

230509 
  walk로 나눠서 RPE 만들고 subgraph 만드는 작업이 있어서, r0~r3으로 나눴던 rpe 값을 벡터로 둠. 
  서브 그래프 생성 후 feature 만들거니까.. (GED 계산을 안할거니까 vector여도 됨!)

  Q. walk 수에 따른 변화



SUREL 의 args
Namespace(B_size=1500, batch_num=2000, batch_size=64, data_usage=1.0, dataset='ogbl-citation2',
 debug=False, directed=False, dropout=0.1, eval_steps=100, gpu_id=0, hidden_dim=64, k=50, l2=0.0,
   layers=2, load_dict=False, load_model=False, log_dir='./log/', lr=0.001, memo=None, metric='mrr',
     model='RNN', norm='all', nthread=16, num_step=4, num_walk=100, optim='adam', patience=5, repeat=1,
       res_dir='./dataset/save', rtest=499, save=False, seed=0, stamp='050923_205618', 
       summary_file='result_summary.log', test_ratio=1.0, train_ratio=0.05, use_degree=False, 
       use_feature=False, use_htype=False, use_val=False, use_weight=False, valid_ratio=0.1, x_dim=0)


'''
from surel_gacc import run_walk
from surel_gacc import run_sample
import numpy as np
from scipy.sparse import csr_matrix

import networkx as nx
import pickle
import sys

from tqdm import tqdm
import time

from operator import itemgetter
# from torch_geometric.utils import subgraph,from_networkx,negative_sampling
import torch

import itertools

import argparse
from cbir_subsg.conf import parse_encoder
from utils import utils
from utils import mkGutils
from surel_gacc import run_walk, run_sample, sjoin
import copy



def gen_batch(iterable, n=1, keep=False):
    length = len(iterable)
    if keep:
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]
    else:
        for ndx in range(0, length - n, n):
            yield iterable[ndx:min(ndx + n, length)]

def np_sampling(rw_dict, ptr, neighs, bsize, target, num_walks, num_steps=3):
    with tqdm(total=len(target)) as pbar:
        for batch in gen_batch(target, bsize, True):
            walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=True)
            node_id, node_freq = freqs[:, 0], freqs[:, 1]
            rw_dict.update(dict(zip(batch, zip(walk_set, node_id, node_freq))))
            pbar.update(len(batch))
    return rw_dict

# path 로 edge list 만들고 edge 추가하기; node path로 Graph 생성 
def mkPathGraph(path):
  # path to edgelist    == [[path[i], path[i+1]] for i in range(len(path)-1)]
  edgelist = list(zip(path, path[1:]))
  G = nx.Graph()
  G.add_edges_from(edgelist)
  return G




# networkx 는 딕셔너리 형태로 그래프를 저장하기 때문에 torch_geometric을 이용해야함
# 노드 간 연결 관계를 표현한 csr_matrix; > 해당 리스트로 nx 를 이용해 graph 생성 가능
def nx2csr(G):
    return csr_matrix(nx.to_scipy_sparse_array(G))

# path 로 edge list 만들고 edge 추가하기; node path로 Graph 생성 
def mkSubGraph(S, K, mF, nodeDict):
  #map이나 mp로 시간 단축 해야함. 일단 구현한다 구현..
  subGList = []
  rpeAggList = []
  cnt = 0
  for idx, sPath in enumerate(S):
     # path to edgelist    == [[path[i], path[i+1]] for i in range(len(path)-1)]
    edgelist = list(zip(sPath, sPath[1:]))
    subG = nx.Graph()
    subG.add_edges_from(edgelist)
    tensor_concat = torch.cat([x.unsqueeze(0) for x in mF[cnt: cnt + len(K[idx])]], dim=0).float()
    enc_agg = torch.mean(tensor_concat, dim=0)

    for i in range(len(K[idx])):
      print("K[idx]: ", K[idx])      
      # subG.nodes[K[idx][i]].update(nodeDict[int(i)]) #F0 attribute 
      try:
        subG.nodes[K[idx][i]].update(nodeDict[K[idx][i]]) #F0 attribute 
      except:
        print("K[idx][i]:", K[idx][i])
        print("nodeDict.keys():", nodeDict.keys())
        print("nodeDict.values():", nodeDict.values())
        print("nodeDict[i]:", nodeDict[K[idx][i]])
        subG.nodes[K[idx][i]].update(nodeDict[str(i)]) #F0 attribute 
        print("i:", i)
        print("type(i):", type(i))
        print("K[idx]:", K[idx])


      subG.nodes[K[idx][i]]['rpe'] = mF[cnt] # rpe값은 tensor임
      cnt+=1
      # subG.nodes[K[idx][i]].update(n)
    subGList.append(subG)
    rpeAggList.append(enc_agg) # 노드의 rpe 값 concat


  return subGList, rpeAggList



#node 각각에 feature 추가할 때 structural feature 로 enc 값 추가해야함
# mk RPE encoding, subgraphs
def mkSubs(G, args, seed ):
  # originGraph의 feature를 가져옴
  nodeDict = dict((x, y ) for x, y in G.nodes(data=True))
  # print("G : ",G)
  subGList, subGFeatList = [], []
  G_full = nx2csr(G)

  print("G_full: ", G_full)


  ptr = G_full.indptr
  neighs = G_full.indices
  num_pos, num_seed, num_cand = len(set(neighs)), 100, 5
  candidates = G_full.getnnz(axis=1).argsort()[-num_seed:][::-1]
  # print("candidates: ", candidates)
  # print("len(candidates): ", len(candidates))

  rw_dict = {}
  B_queues  = []

  # for r in range(1, args.repeat + 1): # 모든 노드에 대해 한 번씩 할거라 repeat 필요 X
  batchIdx, patience = 0, 0
  pools = np.copy(candidates)
  np.random.shuffle(B_queues)
  # while True:
  # if r <= 1:
  B_queues.append(sorted(run_sample(ptr,  neighs, pools, thld=1500))) # pool를 인자로 넣어 모든 노드에 대해 수행하도록 함
  B_pos = B_queues[batchIdx]
  B_w = [b for b in B_pos if b not in rw_dict]
  if len(B_w) > 0:
      walk_set, freqs = run_walk(ptr, neighs, B_w, num_walks=args.num_walks, num_steps=args.num_steps - 1, replacement=True)
  node_id, node_freq = freqs[:, 0], freqs[:, 1]
  rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))
  # else:
  #     if batchIdx >= len(B_queues):
  #         break
  #     else:
  #         B_pos = B_queues[batchIdx]
  batchIdx += 1

  # obtain set of walks, node id and DE (counts) from the dictionary
  S, K, F = zip(*itemgetter(*B_pos)(rw_dict))
  # print("S: ",S[0]): 0번 노드로 시작하는 서브 그래프( walks set ) -> len(walks set): num_walks*(num_step+1)
  # print("K: ",K[0]): 0번 노드로 시작하는 서브 그래프의 .nodes() <- 이때 해당 노드의 F0 값은 없음
  # print("F: ",F[0]): S[0]의 Feature 값; 그리도

  F = np.concatenate(F) #([[[0] * F.shape[-1]], F])   # rpe encoding 값(step 들만)
  mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F]))  #.to(device) # walk를 각 노드에 맞춰서 concat
  gT = mkGutils.normalization(mF, args)

  listA = [a.flatten().tolist() for a in K] 
  flatten_listA = list(itertools.chain(*listA))  # 35*12

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




def main(args):
  # scene graph_원본
  # with open('dataset/v3_x1000.pickle', 'rb') as f:   # time:  74.21744275093079
  with open('data/Vidor/scenegraph/0_2754378442_6188920051.pkl', 'rb') as f:   # time:  74.21744275093079
      data = pickle.load(f)

  # for idx, file in enumerate(data[0]): #graph List List
  #    print("file: ", file)
  #    if len(file)!= 0:      
  #       print(data[0][idx]) #graph  
  #       print(data[1][idx]) #json file name
  #       print(data[2][idx]) #fid

  totalGraph = []

  cnt = 0
  for idx, file in enumerate(data[0]): #graph List List
    #  print("file: ", file)
    if len(file)!= 0:    
      totalGraph.extend(data[0][idx])
  
  # print("len(totalGraph): ", len(totalGraph))

  # with open("data/Vidor/scenegraph/only_upper6nodesGraph.pickle", "wb") as fw:
  #   pickle.dump(totalGraph, fw)

  # sys.exit()


  # num_walks = 4  # num_walks = walk의 총 갯수 
  # num_steps = 3  # num_steps = walk의 길이 = num_steps + 1(start node) 
  pools = 5
  seeds = np.random.choice(pools, 5, replace=False)
  start = time.time()
  # totalData = []
  metaData = [] # 각 originGId 당 생성된 subGList의 개수가 들어감 - originGId, len(subGList)
  totalSubG = []
  totalSubGFeat = []
  for originGId, G in enumerate(tqdm(data)): # 원본 데이터의 오류로 없는 그래프가 종종 있음.try catch 해서 넘기기
    if(len(G.nodes()) != 0):
      subGList, subGFeatList = mkSubs(G, args, seeds)
      metaData.append((originGId, len(subGList)))
      totalSubG+= subGList
      totalSubGFeat += subGFeatList
  end = time.time()

  # print("len(metaData): ",len(metaData))
  # with open('dataset/img100_walk4_step3/walkset_meta.pkl', 'wb') as f:
  #   pickle.dump(metaData, f)

  # print("len(totalSubG): ",len(totalSubG))
  # with open('dataset/img100_walk4_step3/subG.pkl', 'wb') as f:
  #   pickle.dump(totalSubG, f)

  # print("len(totalSubGFeat): ",len(totalSubGFeat))
  # with open('dataset/img100_walk4_step3/subGFeat.pkl', 'wb') as f:
  #   pickle.dump(totalSubGFeat, f)

  # print("len(totalSubGFeat): ",len(totalSubGFeat))

  # end2 = time.time()
  # print("time2: ", end2-start)

  with open('dataset/img100_walk4_step3/subGFeat.pkl', 'rb') as f:
    list2 = pickle.load(f)
  print("len(list2): ",len(list2))
  

# ---------------------------- ^^^ 저장 ^^^ ----------------------------
  # # totalData = {"origin Graph Id ": int, "subGraphs": [(int) subgraphId, (networkx graph) subgraph , (4*4 vector; np array?) sfeature  } }
  # totalData = {}
  # totalData.update(dict(zip(orgGId, zip(subGId, subG, sFeat))))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Embedding arguments')
  utils.parse_optimizer(parser)
  parse_encoder(parser)
  args = parser.parse_args()


  main(args)
  #print("mkGraphRPE")
  