'''
  SUREL에 나온 RPE encoding을 이용해 subraph의 feature를 추출; 
  각 워크는 노드 4개를 가지며, 모든 노드로부터 출발하도록 변경할 필요..?< 왜 모든 노드에 대해 한 게 아니지..?

    freqs[:, 0]: walkset 에 나오는 node id들의 집합
    freqs[:, 1]:  walkset 에 나오는 node id 들의 encoding된 값; 동일한 노드여도 다른 값을 가질 수 있음; 워크마다 다른 구조적 특징값을 가지니까
    => 워크 셋이 아닌 워크로 서브 그래프를 만들고, 해당 서브그래프에 freqs[:, 1]의 enc 값을 특징으로 부여함; 이렇게 해서 subgraph split을 하고,
    이에 대한 ged 계산은 CBIR-SG에서 사용한 astar GED 코드를 사용

0429 1000개 그래프에 대해 서브그래프와 GED 셋 만들고, 
  서브 그래프 자체의 feature에 structural Graph feature를 추가해서 학습 가능하도록 코드 변경하기..


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

def np_sampling(rw_dict, ptr, neighs, bsize, target, num_walks=100, num_steps=3):
    with tqdm(total=len(target)) as pbar:
        for batch in gen_batch(target, bsize, True):
            walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=True)
            node_id, node_freq = freqs[:, 0], freqs[:, 1]
            rw_dict.update(dict(zip(batch, zip(walk_set, node_id, node_freq))))
            pbar.update(len(batch))
    return rw_dict


# 노드 간 연결 관계를 표현한 csr_matrix; > 해당 리스트로 nx 를 이용해 graph 생성 가능ㅁ
def nx2csr(G):
    return csr_matrix(nx.to_scipy_sparse_array(G))

# path 로 edge list 만들고 edge 추가하기; node path로 Graph 생성 
def mkPathGraph(path):
  # path to edgelist    == [[path[i], path[i+1]] for i in range(len(path)-1)]
  edgelist = list(zip(path, path[1:]))
  G = nx.Graph()
  G.add_edges_from(edgelist)
  return G



#node 각각에 feature 추가할 때 structural feature 로 enc 값 추가해야함
# mk RPE encoding, subgraphs
def mkSubs(G, num_walks, num_steps, seeds):
  # originGraph의 feature를 가져옴
  nodeDict = dict((x, y ) for x, y in G.nodes(data=True))
    
  subGList, subGFeatList = [], []
  rw_dict = {}
  rninWalk_dict = {}

  # G=  nx.to_scipy_sparse_array(G) #data[0]: Graph with 35 nodes and 31 edges
  G_full = nx2csr(G)
  ptr = G_full.indptr
  neighs = G_full.indices

  # node 개수에 따라 전체 노드가 아닌 일부만 
  if  len(G.nodes()) > 100:
    B_queues = []
    B_queues.append(sorted(run_sample(G_full.indptr, G_full.indices, seeds, thld=1500))) #그냥 모든 node에 대해서 해도 되는지 확인
  else: 
    B_queues = []
    B_queues.append(list(range(len(G.nodes()))))
  
  batchIdx, patience = 0, 0
  B_pos = B_queues[batchIdx]
  batch = [b for b in B_pos if b not in rw_dict]

  # walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=False)
  # node_id, node_freq = freqs[:, 0], freqs[:, 1]
  if len(batch) > 0 :
    walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=False)
    node_id, node_freq = freqs[:, 0], freqs[:, 1]
    subGList, subGFeatList = [], []
    walksList = []


    for idx, walks in enumerate(walk_set):
      tempWalks = [walks[i:i+num_steps+1] for i in range(0, len(walks), num_steps+1)]
      # todo 이거 병렬적으로 못하나? map 같은거..
      for walk in tempWalks: 
        # subG 생성
        # print("walk: ", walk)
        # print("node_id[idx]: ",node_id[idx]) # 0은 첫번째 워크셋을 의미 
        # 각 노드의 walk set에 있는 node id로 

        featL = []
        subG = mkPathGraph(walk)
        for idx2 in walk:         
          #idx2: node id
          #idx: walkset idx          
          rpe  = node_freq[idx][np.where(node_id[idx] == idx2)][0] #이거 int 형으로 변경해야 사용 가능 -> astar에서 사용 가능한 타입이 int, float, str, none인데 none은 아예 비활임

          featL.extend(rpe)

          attr_list = [f"r{ridx}" for ridx in range(num_steps + 1)]
          for ridx, fname in enumerate (attr_list):   
            subG.nodes[idx2].update({fname : int(rpe[ridx])})

        #subG의 node의 attribute 추가
        #walk == path
        nx.set_node_attributes(subG, nodeDict)

        # print("after attribute attached")
        # print("subG : ", subG)
        # print("subG : ", subG.nodes(data=True))
        if(len(subG))>=4:
          subGList.append(subG)
          subGFeatList.append(featL)
        else: 
          continue
        
    # print(len(subGFeatList))

    # print(chuncked_walks)

    '''
      1. chunck를 만들면서 subgraph 생성, rpe enc 값 concat
  
    '''  
  else:
    if batchIdx >= len(B_queues):
      print("batchIdx >= len(B_queues)")
      print("B_queues: ", B_queues)
    else:
      B_pos = B_queues[batchIdx]
  # walk set을 walk_step+1 길이로 나눠서 서브 그래프 생성
  return subGList, subGFeatList


def main():
  # scene graph_원본
  with open('dataset/v3_x1000.pickle', 'rb') as f:   # time:  74.21744275093079
      data = pickle.load(f)
  # mk dataset
  # data = data[:1000]
  orgGIdList = []
  for orgGId in range(len(data)):
      orgG = data[orgGId]
      orgGIdList.append(orgGId)
      subList = []

  num_walks = 1  # walk의 총 갯수 
  num_steps =  4  # walk의 길이 = num_steps + 1(start node) 
  pools = 5
  seeds = np.random.choice(pools, 5, replace=False)

  start = time.time()
  # totalData = []
  metaData = [] # 각 originGId 당 생성된 subGList의 개수가 들어감 - originGId, len(subGList)
  totalSubG = []
  totalSubGFeat = []
  for originGId, G in enumerate(tqdm(data)): # 원본 데이터의 오류로 없는 그래프가 종종 있음.try catch 해서 넘기기
    try:
      #  print("idx: ",idx)
      subGList, subGFeatList = mkSubs(G, num_walks, num_steps, seeds)
      metaData.append([originGId, len(subGList)])
      totalSubG += subGList
      totalSubGFeat += subGFeatList
    except:
      continue

  end = time.time()
  print("time1 : ", end-start)



  print("len(metaData): ",len(metaData))
  with open('dataset/rpe_splited/rpe_v3_x1000_step1_meta.pkl', 'wb') as f:
    pickle.dump(metaData, f)

  print("len(totalSubG): ",len(totalSubG))
  with open('dataset/rpe_splited/rpe_v3_x1000_step1_SubG.pkl', 'wb') as f:
    pickle.dump(totalSubG, f)

  print("len(totalSubGFeat): ",len(totalSubGFeat))
  
  with open('dataset/rpe_splited/rpe_v3_x1000_SubGFeat.pkl', 'wb') as f:
    pickle.dump(totalSubGFeat, f)

  end2 = time.time()
  print("time2: ", end2-start)

  with open('dataset/rpe_splited/rpe_v3_x1000_step1_SubG.pkl', 'rb') as f:
    list2 = pickle.load(f)
  print("len(list2): ",len(list2))
  

  # sub Grpah 와 meta data 따로 생성; pickle 파일이 커서 오류 발생
  # 데이터 표현하는 방법만 차용할거면 sugG_acc 자체의 코드를 변경해서 데이터 생성 시간을 개선하는게 나아보임. 
  # subgraph마다 node id에 따라 feature를 부여해야함.. -> 애초에 만들어놓은 게 노드 id에서 node 이름으로 feature 부여하는 코드 있지 않았나? 
  # 데이터 생성 시 node class feature 를 추가하는 것과 추가하지 않는 경우의 시간차이도 비교해야함..

  # # totalData = {"origin Graph Id ": int, "subGraphs": [(int) subgraphId, (networkx graph) subgraph , (4*4 vector; np array?) sfeature  } }
  # totalData = {}
  # totalData.update(dict(zip(orgGId, zip(subGId, subG, sFeat))))



if __name__ == "__main__":
  
  main()
  #print("mkGraphRPE")

