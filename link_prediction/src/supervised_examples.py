import time
import os
from datetime import datetime
import random
import pickle
import torch

from data_process import *

# supervised, because we use ground truth, malicious benign

# extract communicating pairs of nodes
# we are not interested in how many edges or types of edges here
def construct_pos(path, ground_truth, posSrc=False):
    
    pos_examples = {}  
    pos_src = {}

    f = open(path, 'r')
    print("\nConstructing positive examples from: ", path)
    
    for line in f:
        temp = line.strip('\n').split('\t')
        if temp[0] in ground_truth and temp[2] in ground_truth:
            pair = (temp[0], temp[2])
            if pair in pos_examples: continue
            pos_examples[pair] = 1

            if not posSrc: continue

            if temp[0] not in pos_src:
                pos_src[temp[0]] = set()
            pos_src[temp[0]].add(temp[2])

    f.close()
    print("selected_pos: ", len(pos_examples))

    return pos_examples, pos_src


# negative link examples have a malicious source node (alert); for the destination, we sample from the graph nodes they never connect to
# node_mappings: id -> list_of_adj_ids
# uuid_to_id: uuid -> id
# id_to_uuid: id -> uuid

def construct_neg_helper(path, ground_truth, uuid_to_id, id_to_uuid, node_adj, sample_size=1, negSrc=False):
    
    nodes_num = len(id_to_uuid)

    f = open(path, 'r')
    print("Extracting negatives from: ", path)
    neg_examples = {} 
    srcs = set()
    neg_src = {}

    for line in f:
        temp = line.strip('\n').split('\t')
        src = temp[0]
        if src not in ground_truth: continue
        
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
        
        if not negSrc: continue

        # construct a few neg examples per src
        if src in neg_src: continue
        neg_src[src] = set()
        samples = random.sample(range(0, nodes_num), 10)
        for i in samples:
            if i in node_adj[src_index]: continue # this link exists, can't use it as a negative example
            neg_src[src].add(id_to_uuid[i])

    f.close()

    return neg_examples, neg_src


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
        

def construct_neg(myfile, ground_truth, sample_size=1, negSrc=False):
    # we need the node mappings to construct negative examples for training
    uuid_to_id={}
    id_to_uuid={}
    node_adj={}

    print("\nConstructing negative examples from: ", myfile)

    # add mappings from campaign training file
    uuid_to_id, id_to_uuid, node_adj = node_mappings(myfile, uuid_to_id, id_to_uuid, node_adj)

    # construct negative examples for training
    neg_examples, neg_src = construct_neg_helper(myfile, ground_truth, uuid_to_id, id_to_uuid, node_adj, sample_size=sample_size, negSrc=negSrc)
    print("neg examples: ", len(neg_examples))

    return neg_examples, neg_src

    
# mapping to global node ids
def format_examples(examples, nodeId_map):
    examples = list(examples.keys())
    examples_first, examples_second = zip(*examples)

    examples_first_id = [nodeId_map[uuid] for uuid in examples_first]
    examples_second_id = [nodeId_map[uuid] for uuid in examples_second]

    return examples_first_id, examples_second_id, examples_first, examples_second


def get_ground_truth(path_list):
    ground_truth = {}
    for path in path_list:
        f_gt = open(path, 'r')
        for line in f_gt:
            ground_truth[line.strip('\n')] = 1
        f_gt.close()

    return ground_truth


def construct_examples_supervised(scenarios, path_dir, nodeType_map, edgeType_map, posSrc=False, negSrc=False):

    data_split = {}
    # path_dir = "../data/E3/graph_data"
    path_ground_truth = "../data/E3/groundtruth"
    path_examples = "../examples"

    # positive examples - direct link exist
    # negative examples - no direct link exists
    pos_first = []
    pos_second = []
    neg_first = []
    neg_second = []
    
    pos_uuid_first = []
    pos_uuid_second = []
    neg_uuid_first = []
    neg_uuid_second = []

    # only interested in these for the test scenario
    pos_src = {}
    neg_src = {}

    provenance, nodeId_map = index_dataset(scenarios, path_dir, nodeType_map, edgeType_map)

    # ground truth (malicious)
    ground_truth_file_list = []
    for scenario in scenarios:
        ground_truth_file_list.append(os.path.join(path_ground_truth, "{}.txt".format(scenario)))
    ground_truth = get_ground_truth(ground_truth_file_list)
    
    for scenario in scenarios:
        mypath_attack = os.path.join(path_dir, scenario, scenario + "_test.txt")

        pos, pos_src = construct_pos(mypath_attack, ground_truth, posSrc=posSrc)
        pos_first_tmp, pos_second_tmp, pos_uuid_first_tmp, pos_uuid_second_tmp = format_examples(pos, nodeId_map)

        neg, neg_src = construct_neg(mypath_attack, ground_truth, sample_size=1, negSrc=negSrc)
        neg_first_tmp, neg_second_tmp, neg_uuid_first_tmp, neg_uuid_second_tmp = format_examples(neg, nodeId_map)
    
        pos_first += pos_first_tmp
        pos_second += pos_second_tmp

        neg_first += neg_first_tmp
        neg_second += neg_second_tmp
        
        pos_uuid_first += pos_uuid_first_tmp
        pos_uuid_second += pos_uuid_second_tmp

        neg_uuid_first += neg_uuid_first_tmp
        neg_uuid_second += neg_uuid_second_tmp

    size = min(len(pos_first), len(neg_first)) # use the same number of pos/neg examples for balance
    
    pos_first = pos_first[:size]
    pos_second = pos_second[:size]
    neg_first = neg_first[:size]
    neg_second = neg_second[:size]
    
    pos_uuid_first = pos_uuid_first[:size]
    pos_uuid_second = pos_uuid_second[:size]
    neg_uuid_first = neg_uuid_first[:size]
    neg_uuid_second = neg_uuid_second[:size]

    data_split["pos"] = [pos_first, pos_second]
    data_split["neg"] = [neg_first, neg_second]
    data_split["pos_uuid"] = [pos_uuid_first, pos_uuid_second]
    data_split["neg_uuid"] = [neg_uuid_first, neg_uuid_second]

    # save examples to file
    scenario_str = scenarios[0]
    if len(scenarios) > 1:
        for scenario in scenarios[1:]:
            scenario_str += '_' + scenario
    
    filename_examples = 'examples_uuid.pickle'

    path_examples = os.path.join(path_examples, scenario_str)
    os.makedirs(path_examples, exist_ok = True)
    filename_examples = os.path.join(path_examples, filename_examples)
    filename_pos_src = os.path.join(path_examples, 'pos_src.pickle')
    filename_neg_src = os.path.join(path_examples, 'neg_src.pickle')
    filename_nodeId_map = os.path.join(path_examples, 'nodeId_map.pickle')

    print("Saving {} examples to {}".format(len(data_split["pos"][0]), filename_examples))
    
    with open(filename_examples, 'wb') as handle:
        pickle.dump(data_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Examples have been saved in: ", filename_examples)

    with open(filename_nodeId_map, 'wb') as handle:
        pickle.dump(nodeId_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("NodeId map has been saved in: ", filename_nodeId_map)

    if pos_src and neg_src:
        with open(filename_pos_src, 'wb') as handle:
            pickle.dump(pos_src, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pos src have been saved in: ", filename_pos_src)

        with open(filename_neg_src, 'wb') as handle:
            pickle.dump(neg_src, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Neg src have been saved in: ", filename_neg_src)



def format_examples_2(pos_src, neg_src, nodeId_map, uuid):
    data_split = {}
    pos_vals = list(pos_src[uuid])
    neg_vals = list(neg_src[uuid]) 

    pos_first = [nodeId_map[uuid] for i in range(len(pos_vals))]
    pos_second = [nodeId_map[j] for j in pos_vals]
    
    neg_first = [nodeId_map[uuid] for i in range(len(neg_vals))]
    neg_second = [nodeId_map[j] for j in neg_vals]

    data_split["pos"] = [pos_first, pos_second]
    data_split["neg"] = [neg_first, neg_second]

    return data_split


def load_examples_supervised(scenarios, filename_examples, posSrc=False, negSrc=False, index=10):

    path_examples = "../examples"
    pos_src = {}
    neg_src = {}
    data_split_pos_neg = {}
    
    scenario_str = scenarios[0]
    if len(scenarios) > 1:
        for scenario in scenarios[1:]:
            scenario_str += '_' + scenario

    path_examples = os.path.join(path_examples, scenario_str)
    filename_examples = os.path.join(path_examples, filename_examples)
    filename_nodeId_map = os.path.join(path_examples, 'nodeId_map.pickle')
    filename_pos_src = os.path.join(path_examples, 'pos_src.pickle')
    filename_neg_src = os.path.join(path_examples, 'neg_src.pickle')

    with open(filename_examples, 'rb') as handle:
        data_split = pickle.load(handle)

    if posSrc and negSrc:
        with open(filename_nodeId_map, 'rb') as handle:
            nodeId_map = pickle.load(handle)
        
        with open(filename_pos_src, 'rb') as handle:
            pos_src = pickle.load(handle)
        
        with open(filename_neg_src, 'rb') as handle:
            neg_src = pickle.load(handle)

        uuid = list(pos_src.keys())[index]

        data_split_pos_neg = format_examples_2(pos_src, neg_src, nodeId_map, uuid)

    return data_split, pos_src, neg_src, data_split_pos_neg


#if __name__ == "__main__":
    #construct_pos_neg(scenarios = ["trace"]) # need nodeId_map


