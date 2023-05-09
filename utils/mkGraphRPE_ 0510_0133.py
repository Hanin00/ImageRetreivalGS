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
Namespace(B_size=1500, batch_num=2000, batch_size=32, data_usage=1.0, dataset='ogbl-citation2',
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




from operator import itemgetter
from torch_geometric.utils import subgraph,from_networkx,negative_sampling
import torch

from config.config import parse_encoder
import argparse
from utils import utils
from utils import mkGutils

from surel_gacc import run_walk, run_sample, sjoin

import copy


# networkx 는 딕셔너리 형태로 그래프를 저장하기 때문에 torch_geometric을 이용해야함

# 노드 간 연결 관계를 표현한 csr_matrix; > 해당 리스트로 nx 를 이용해 graph 생성 가능ㅁ
def nx2csr(G):
    return csr_matrix(nx.to_scipy_sparse_array(G))



#node 각각에 feature 추가할 때 structural feature 로 enc 값 추가해야함
# mk RPE encoding, subgraphs
def mkSubs(G, args, seeds, ):
  # originGraph의 feature를 가져옴
  nodeDict = dict((x, y ) for x, y in G.nodes(data=True))

  subGList, subGFeatList = [], []
  Gk = copy.deepcopy(G)


  G_full = nx2csr(G)
  # G=  nx.to_scipy_sparse_array(G) #data[0]: Graph with 35 nodes and 31 edges
  Gk = from_networkx(Gk)
  print("G. edge_index ----------------")
  print(Gk.edge_index)
  
  ptr = G_full.indptr
  neighs = G_full.indices
  num_pos, num_seed, num_cand = len(set(neighs)), 100, 5
  candidates = G_full.getnnz(axis=1).argsort()[-num_seed:][::-1]

  T_edge_idx = Gk.edge_index
  F_edge_idx = negative_sampling(edge_index=Gk.edge_index, num_nodes=Gk.num_nodes)


  print("T_edge_idx: ",T_edge_idx)
  print("F_edge_idx: ",F_edge_idx)


  rw_dict = {}
  B_queues  = []

  r = 1
  
  # node 개수에 따라 전체 노드가 아닌 일부만 
  if  len(G.nodes()) > 100:
    B_queues = []
    B_queues.append(sorted(run_sample(ptr, neighs, seeds, thld=1500)))
    # B_queues.append(sorted(run_sample(G_full.indptr, G_full.indices, seeds, thld=1500)))
  else: 
    B_queues = []
    B_queues.append(list(range(len(G.nodes()))))

  # while True:
  #     if r <= 2:
  batchIdx, patience = 0, 0
  B_pos = B_queues[batchIdx]
  batch = [b for b in B_pos if b not in rw_dict]

  if len(batch) > 0:
    walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=args.num_walks, num_steps= args.num_steps - 1, replacement=True)
    node_id, node_freq = freqs[:, 0], freqs[:, 1]
    rw_dict.update(dict(zip(batch, zip(walk_set, node_id, node_freq))))
  else:
    if batchIdx >= len(B_queues):
        print("BatfchIdx >= len(B_queues)")
      # break
    else:
      B_pos = B_queues[batchIdx]
  batchIdx += 1





  S, K, F = zip(*itemgetter(*B_pos)(rw_dict))

  S = torch.from_numpy(np.asarray(S)).long()  #  num_samples * num_walks; walks set 
  # 각 노드로부터 시작하는 Randomwalk set

  F = np.concatenate(F)
  F = np.concatenate([[[0] * F.shape[-1]], F])   # rpe encoding 값(step 들만)
  mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F])) # train.py
  mF = torch.from_numpy(F) #    ; tensor; main.py  ; rpe encoding을 연결하면 각 노드를 연결하여 두 그래프가 공통으로 가지고 있는 노드에 대해 연결해 하나로 표현할 수 있음
  #print("mF: ",mF)

  # print("S: ",S) # 
  # print("K: ", K) # walkset 에 있는 node ID ;

  gT = mkGutils.normalization(mF, args) #normalized

  print("S: ", S)
  print("K: ", K)
  print("F: ", F)
  print("------------------")


  print("gT: ", gT)


#   #loss, auc = train(model, optimizer, data, gT)



# # obtain set of walks, node id and DE (counts) from the dictionary
#   S, K, F = zip(*itemgetter(*B_pos)(rw_dict))




#   # SUREPL/main.py
#   B_pos_edge, _ = subgraph(list(B_pos), T_edge_idx)
#   B_full_edge, _ = subgraph(list(B_pos), F_edge_idx)
#   #data = gen_sample(np.asarray(S), B_pos, K, B_pos_edge, B_full_edge, inf_set['X'], args, gtype=g_class.gtype)
#   # F = np.concatenate(F)   # 각 step의 rpe 생성
  
#   # mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F])) 

# # SUREL/train.py
#   S = torch.from_numpy(np.asarray(S)).long()  #  num_samples * num_walks; walks set 
#   # 각 노드로부터 시작하는 Randomwalk set

#   F = np.concatenate(F)
#   F = np.concatenate([[[0] * F.shape[-1]], F])   # rpe encoding 값(step 들만)
#   mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F])) # train.py
#   mF = torch.from_numpy(F) #    ; tensor; main.py  ; rpe encoding을 연결하면 각 노드를 연결하여 두 그래프가 공통으로 가지고 있는 노드에 대해 연결해 하나로 표현할 수 있음
#   #print("mF: ",mF)

#   # print("S: ",S) # 
#   # print("K: ", K) # walkset 에 있는 node ID ;

#   uvw, uvx = sjoin(S, K, batch, return_idx=True)
#   # print("uvw: ",uvw)

#   # print("uvw[0]",uvw[0])
#   # print("uvw[1]",uvw[1])
  
#   uvw = uvw.reshape(2, -1, 2)


#   x = torch.from_numpy(uvw)
#   gT = mkGutils.normalization(mF, args) #normalized

#   # print(type(gT))
#   # print(gT)

#   # print("gT[uvw[0]]: ",gT[uvw[0]]) 
#   # print("gT[uvw[1]]: ",gT[uvw[1]]) 
#   gT = torch.stack([gT[uvw[0]], gT[uvw[1]]])  # 
        



  


  # # walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=False)
  # # node_id, node_freq = freqs[:, 0], freqs[:, 1]
  # if len(batch) > 0 :
  #   walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=args.num_walks, num_steps=args.num_steps, replacement=False)
  #   node_id, node_freq = freqs[:, 0], freqs[:, 1]
  #   subGList, subGFeatList = [], []

  #   walksList = []
  #   for idx, walks in enumerate(walk_set):
  #     tempWalks = [walks[i:i+num_steps+1] for i in range(0, len(walks), num_steps+1)]
  #     # todo 이거 병렬적으로 못하나? map 같은거..
  #     for walk in tempWalks: 
  #       # subG 생성
  #       # print("walk: ", walk)
  #       # print("node_id[idx]: ",node_id[idx]) # 0은 첫번째 워크셋을 의미 
  #       # 각 노드의 walk set에 있는 node id로 


  #       featL = []
  #       subG = mkPathGraph(walk)
  #       for idx2 in walk:         
  #         #idx2: node id
  #         #idx: walkset idx          
  #         rpe  = node_freq[idx][np.where(node_id[idx] == idx2)][0] #이거 int 형으로 변경해야 사용 가능 -> astar에서 사용 가능한 타입이 int, float, str, none인데 none은 아예 비활임
  #         featL.extend(rpe)
  #         subG.nodes[idx2]['rpe'] = rpe
         
  #         attr_list = [f"r{ridx}" for ridx in range(args.num_steps + 1)]
  #         for ridx, fname in enumerate (attr_list):   
  #           subG.nodes[idx2].update({fname : int(rpe[ridx])})

  #       #subG의 node의 attribute 추가
  #       #walk == path
  #       nx.set_node_attributes(subG, nodeDict)

  '''
    1. chunck를 만들면서 subgraph 생성, rpe enc 값 concat

  '''  
  # else:
  #   if batchIdx >= len(B_queues):
  #     print("batchIdx >= len(B_queues)")
  #     print("B_queues: ", B_queues)
  #   else:
  #     B_pos = B_queues[batchIdx]
  # walk set을 walk_step+1 길이로 나눠서 서브 그래프 생성
  return subGList, subGFeatList


















def main(args):
  # scene graph_원본
  with open('dataset/v3_x1000.pickle', 'rb') as f:   # time:  74.21744275093079
      data = pickle.load(f)
  # mk dataset
  data = data[:100]

  orgGIdList = []
  for orgGId in range(len(data)):
      orgG = data[orgGId]
      orgGIdList.append(orgGId)
      subList = []

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
    subGList, subGFeatList = mkSubs(G, args, seeds)
    try:
      #  print("idx: ",idx)
      subGList, subGFeatList = mkSubs(G, args, seeds)
      metaData.append([originGId, len(subGList)])
      totalSubG += subGList
      totalSubGFeat += subGFeatList
    except:
      continue

  end = time.time()
  print("time1 : ", end-start)


# ---------------------------- vvv 저장 vvv ----------------------------

  print("len(metaData): ",len(metaData))
  with open('dataset/img100_walk4_step2/walkset_meta.pkl', 'wb') as f:
    pickle.dump(metaData, f)

  print("len(totalSubG): ",len(totalSubG))
  with open('dataset/img100_walk4_step2/walkset.pkl', 'wb') as f:
    pickle.dump(totalSubG, f)

  # print("len(totalSubGFeat): ",len(totalSubGFeat))
  
  # with open('dataset/rpe_splited_subgraph/rpe_v3_x1000_walks4_step3_SubGFeat.pkl', 'wb') as f:
  #   pickle.dump(totalSubGFeat, f)

  # end2 = time.time()
  # print("time2: ", end2-start)

  # with open('dataset/rpe_splited_subgraph/rpe_v3_x1000_walks4_step3_SubG.pkl', 'rb') as f:
  #   list2 = pickle.load(f)
  # print("len(list2): ",len(list2))
  


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