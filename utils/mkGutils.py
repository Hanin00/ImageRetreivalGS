import numpy as np
import networkx as nx
from copy import deepcopy
import pickle
import random

from surel_gacc import sjoin

import torch

#SUREL - https://github.com/Graph-COM/SUREL/blob/5c209a4e565440b20371dca4350717e930ffec8b/utils.py#L66
def normalization(T, args):
    if args.use_weight:
        norm = torch.tensor([args.num_walks] * args.num_steps + [args.w_max], device=T.device)
    else:
        if args.norm == 'all':
            norm = args.num_walks
        elif args.norm == 'root':
            norm = torch.tensor([args.num_walks] + [1] * args.num_steps, device=T.device)
        else:
            raise NotImplementedError
    return T / norm


# with Target GEV



def graph_generation(graph, global_labels, global_edge_labels, total_ged=0):
    new_g = deepcopy(graph)
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

    ## edit node labels
    to_edit_idx_newg = random.sample(new_g.nodes(), target_ged['nc'])
    for idx in to_edit_idx_newg:
        while (True):
            toassigned_new_nodetype = random.choice(list(global_labels))
            if (toassigned_new_nodetype != new_g.nodes()[idx]['type']):
                break
        new_g.nodes()[idx]['type'] = toassigned_new_nodetype

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
            if (toassigned_new_edgetype != new_g.edges()[idx]['type']):
                break
        new_g.edges()[idx]['type'] = toassigned_new_edgetype

    ## edit node insertions
    for num in range(target_ged['in']):
        curr_num_node = new_g.number_of_nodes()
        to_insert_edge = random.sample(new_g.nodes(), 1)[0]
        new_g.add_node(str(curr_num_node), label=str(curr_num_node), type=random.choice(global_labels))
        # add edge to the newly inserted ndoe

        
        new_g.add_edge(str(curr_num_node), to_insert_edge, type=random.choice(global_edge_labels))

    ## edit edge insertions
    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        while (True):
            curr_pair = random.sample(new_g.nodes(), 2)
            if ((curr_pair[0], curr_pair[1]) not in deleted_edges):
                # print('poten edge', curr_pair[0], curr_pair[1])
                if ((curr_pair[0], curr_pair[1]) not in new_g.edges()):
                    # print('added adge', curr_pair[0], curr_pair[1])
                    new_g.add_edge(curr_pair[0], curr_pair[1], type=random.choice(global_edge_labels))
                    break

    return new_g, target_ged, 






def load_generated_graphs(dataset_name, file_name='generated_graph_500'):
    dir = 'dataset/' + dataset_name + '/' + file_name
    g = open(dir, 'rb')
    generated_graphs = pickle.load(g)
    g.close()
    return generated_graphs



import sys
# based RPE walkset -> concatenation -> subgraph
def mkSubgraph(Wu, Wv, ):
    ''' 
      walk 두 개를 concat ->  각 노드에는 f0 말고 특징값이 없게 되는데..? 
    '''

    subG = [Wu.nodes['RPE']]
  
    # print(Wu.edges(data=True))    
    # print(Wv.edges(data=True))    

    print(Wu.nodes(data=True))    
    print(Wv.nodes(data=True))    

    subG = nx.Graph()
    subG.add_nodes_from(Wu.nodes(data=True))
    subG.add_edges_from(Wu.edges(data=True))


    print("1: ", subG.nodes(data=True))
    print("1: ", subG.edges(data=True))

    
    subG.add_nodes_from(Wv.nodes(data=True))
    subG.add_edges_from(Wv.edges(data=True))

    print("add Wu")
    print("2: ", subG.nodes(data=True))    
    print("2: ", subG.edges(data=True)) 

    return 



import pickle

def main():

  with open('dataset/img100_walk4_step2/walkset.pkl', 'rb') as f:
    list2 = pickle.load(f)
  print("len(list2): ",len(list2))

  Wu = list2[15]
  Wv = list2[10]

  subG = mkSubgraph(Wu, Wv)
 

if __name__ == "__main__":
    main()

#node type의 class들 -> name, feature를 전에 만들어놓은 dict를 이용해서 넣을 것; 동일 그래프 내의 node로만 생성하거나, 전체 node에 대해 생성(우선)