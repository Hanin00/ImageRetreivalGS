from collections import defaultdict, Counter

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import torch
import torch.optim as optim
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
import networkx as nx
import numpy as np
import random
import scipy.stats as stats
from tqdm import tqdm

import pickle

import sys


def sample_neigh(graphs, size):
    ps = np.array([len(g) for g in graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh
cached_masks = None


def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    #v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    #v = [np.sum(v) for mask in cached_masks]
    return v


def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=np.int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=np.int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                                         axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))


def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
                                       progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size:
                total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
                                     reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0/(k+1)) ** 1.5
    #ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts


def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
                                  node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
                     not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            # if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
                        node_anchored)
        sg.remove(w)


def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
                                 node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    #sizes = {}
    # for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
                                     reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

device_cache = None

def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda:0") if torch.cuda.is_available() \
            else torch.device("cpu")
        if torch.cuda.is_available():
            print("cuda available")
            torch.cuda.empty_cache()
        #device_cache = torch.device("cpu")
    return device_cache

def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.')


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
                              weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



def batch_nx_graphs(graphs, anchors=None):
    # motifs_batch = [pyg_utils.from_networkx(
    #    nx.convert_node_labels_to_integers(graph)) for graph in graphs]
    #loader = DataLoader(motifs_batch, batch_size=len(motifs_batch))
    #for b in loader: batch = b
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])
                
    for g in graphs:
        for v in g.nodes:
            rpe = g.nodes[v]['rpe']
            f0 = g.nodes[v]["txtemb"]
            g.nodes[v]["node_feature"] = torch.tensor(np.concatenate((rpe, f0), axis=None))

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = batch.to(get_device())
    # print(batch)
    return batch

#DSGraph로 변경하는 과정에서 변수명이 key에 없어 된지 않음 -> edge 삭제 후 재생성 시, feature를 변경해서 입력해봄
def batch_nx_graphs_rpe(graphs, anchors=None):
    newGraphs = []
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])
                print("g.nodes[v] : ",  g.nodes[v])
            for e in g.edges:
                g.edges[e]["edge_feature"] = torch.tensor([float(v == anchor)])
                print("g.edges[v] : ",  g.edges[v])
        
    # print("utils - graphs: ",graphs)
    for g in graphs:
        # print("utils - g: ",g)
        newG = nx.Graph()
        newG.add_nodes_from(g.nodes(data=True))
        newG.add_edges_from(g.edges())


        for v in list(g.nodes):
                rpe = g.nodes[v]['rpe']
                f0 = g.nodes[v]["txtemb"]
                newG.nodes[v]["node_feature"] = torch.tensor(np.concatenate((rpe, f0), axis=None))

        for e in list(g.edges):
                txtemb = g.edges[e[0], e[1]]['txtemb'] # 10 
                distance = g.edges[e[0], e[1]]["distance"] #1
                angle_AB = g.edges[e[0], e[1]]["angle_AB"] # 1
                angle_BA = g.edges[e[0], e[1]]["angle_BA"] #1 
                newG.edges[e]["edge_feature"] = torch.tensor(np.concatenate((txtemb, distance,angle_AB,angle_BA), axis=None))
        newGraphs.append(newG)
        
    # print("utils - newGraphs: ",newGraphs)
    

    batch = Batch.from_data_list([DSGraph(g) for g in newGraphs])  

    try:
        batch = batch.to(get_device())
    except:
        print(graphs)
    # print(batch)
    return batch



# feature_extract_scenegraph.py에서 가져옴



from utils.mkGraphRPE import *
from surel_gacc import run_walk

def mkMergeGraph(S, K, gT, nodeNameDict, F0dict, nodeIDDict):

    merged_K = np.concatenate([np.asarray(k) for k in K]).tolist()
    # print("merged_K: ",merged_K)
    merged_K = [nodeIDDict[i] for i in merged_K]
    
    sum_dict = {}
    count_dict = {}
    for k, gf in zip(merged_K, gT):
        if k in sum_dict:
            count_dict[k] += 1
        else:
            sum_dict[k] = gf
            count_dict[k] = 1 
    gT_mean = {k: sum_dict[k] / count_dict[k] for k in sum_dict} #
    
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
 

#-----^^^^ RPE 계산 ^^^^-------------------