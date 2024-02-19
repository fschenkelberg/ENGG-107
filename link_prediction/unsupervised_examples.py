import time
import os
from datetime import datetime
import random
import pickle
import torch

# unsupervised, because we do not use ground truth
# the "labels" only denote link existence; possible extension to random walks


# extract communicating pairs of nodes
# we are not interested in how many edges or types of edges here
def construct_pos(path):
    
    pos_examples = {}  
    
    f = open(path, 'r')
    print("\nConstructing positive examples from: ", path)
    
    for line in f:
        temp = line.strip('\n').split('\t')
        pair = (temp[0], temp[2])
        if pair in pos_examples: continue
        pos_examples[pair] = 1

    f.close()
    print("selected_pos: ", len(pos_examples))

    return pos_examples


# negative link examples are betwwen two nodes that do not communicate
# node_mappings: id -> list_of_adj_ids
# uuid_to_id: uuid -> id
# id_to_uuid: id -> uuid

def construct_neg_helper(path, uuid_to_id, id_to_uuid, node_adj, sample_size=1):
    
    nodes_num = len(id_to_uuid)

    f = open(path, 'r')
    print("Extracting negatives from: ", path)
    neg_examples = {} 
    srcs = set()

    for line in f:
        temp = line.strip('\n').split('\t')
        src = temp[0]
        
        # process each node once
        #if src in srcs: continue
        #srcs.add(src)

        src_index = uuid_to_id[src]

        # sample from node_ids
        samples = random.sample(range(0, nodes_num), sample_size)
        for i in samples:
            if i in node_adj[src_index]: continue # this link exists, can't use it as a negative example
            pair = (src, id_to_uuid[i])
            neg_examples[pair] = 0

    f.close()

    return neg_examples


# just some auxiliary mapping functions to identify negative examples
def node_mappings(mypath, uuid_to_id={}, id_to_uuid={}, node_adj={}):
    index = len(id_to_uuid)

    print("Extracting node mappings from: ", mypath)
    f = open(mypath, 'r')
    for line in f:
        temp = line.strip('\n').split('\t')
        src = temp[0]
        dest = temp[2]
        if src not in uuid_to_id:
            uuid_to_id[src] = index
            id_to_uuid[index] = src
            node_adj[index] = set()
            index += 1
        if dest not in uuid_to_id:
            uuid_to_id[dest] = index
            id_to_uuid[index] = dest
            node_adj[index] = set()
            index += 1
        src_index = uuid_to_id[src]
        dest_index = uuid_to_id[dest]
        node_adj[src_index].add(dest_index)

    f.close()

    return uuid_to_id, id_to_uuid, node_adj
        

def construct_neg( myfile, sample_size=1):
    # we need the node mappings to construct negative examples for training
    uuid_to_id={}
    id_to_uuid={}
    node_adj={}

    print("\nConstructing negative examples from: ", myfile)

    # add mappings from campaign training file
    uuid_to_id, id_to_uuid, node_adj = node_mappings(myfile, uuid_to_id, id_to_uuid, node_adj)

    # construct negative examples for training
    neg_examples = construct_neg_helper(myfile, uuid_to_id, id_to_uuid, node_adj, sample_size=sample_size)
    print("neg examples: ", len(neg_examples))

    return neg_examples

    
# mapping to global node ids
def format_examples(examples, nodeId_map):
    examples = list(examples.keys())
    examples_first, examples_second = zip(*examples)

    examples_first = [nodeId_map[uuid] for uuid in examples_first]
    examples_first = torch.tensor(examples_first, dtype=torch.long)
    
    examples_second = [nodeId_map[uuid] for uuid in examples_second]
    examples_second = torch.tensor(examples_second, dtype=torch.long)

    return examples_first, examples_second


def construct_examples_unsupervised(scenarios = ["trace"], nodeId_map={}):

    data_split = {}
    path_dir = "../data/E3/graph_data"

    # positive examples - direct link exist
    # negative examples - no direct link exists
    pos_first = []
    pos_second = []
    neg_first = []
    neg_second = []
    
    for scenario in scenarios:
        mypath_attack = os.path.join(path_dir, scenario, scenario + "_test.txt")

        pos = construct_pos(mypath_attack)
        pos_first_tmp, pos_second_tmp = format_examples(pos, nodeId_map)
        pos_first += pos_first_tmp
        pos_second += pos_second_tmp

        neg = construct_neg(mypath_attack, sample_size=1)
        neg_first_tmp, neg_second_tmp = format_examples(neg, nodeId_map)
        neg_first += neg_first_tmp
        neg_second += neg_second_tmp

    size = min(len(pos_first), len(neg_first)) # use the same number of pos/neg examples for balance
    data_split["pos"] = [pos_first[:size], pos_second[:size]]
    data_split["neg"] = [neg_first[:size], neg_second[:size]]

    return data_split


#if __name__ == "__main__":
    #construct_pos_neg(scenarios = ["trace"]) # need nodeId_map


