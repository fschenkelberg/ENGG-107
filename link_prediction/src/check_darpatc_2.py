import pickle
import os
import random
import numpy as np
import torch
import operator
from data_process import *


def get_freq_node():

    seed = 5 # 1 and 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    h_feats = 256 # embeddings dim
    lr = 0.001  # learning rate
    epochs = 300
    seed_emb = 5

    print("\nUsing GraphSage Embeddings with Parameters:")

    param_str = 'dim' + str(h_feats) + '_seed' + str(seed_emb) + '_lr' + str(lr)[2:] + '_epochs' + str(epochs)
    print("Embeddings Size: ", h_feats)
    print("Embeddings Training Epochs: ", epochs)
    print("Embeddings Model (GNNs) Learning Rate: ", lr)

    path_emb = "../embeddings/train_on_trace_fivedirections_cadets/" + param_str
    #path_emb = "../embeddings/train_on_trace/" + param_str

    files_emb = {'cadets': os.path.join(path_emb, 'emb_cadets.pickle'),
                 'trace': os.path.join(path_emb, 'emb_trace.pickle'),
                 'theia': os.path.join(path_emb, 'emb_theia.pickle'),
                 'fivedirections': os.path.join(path_emb, 'emb_fivedirections.pickle')}

    scenario_list = ['cadets'] # all
    #scenario_list = ['theia'] # all
    #scenario_list = ['trace'] # all

    freq_cadets = 28
    freq_theia = 11422
    freq_trace = 3
    freq = freq_cadets

    for scenario in scenario_list:
        filename_emb = files_emb[scenario]
        print("Loading embeddings from: ", filename_emb)
        with open(filename_emb, 'rb') as handle:
            ecrt = pickle.load(handle)
            #embeddings_mat = torch.cat([embeddings_mat, ecrt])
            print("\nEmbeddings:")
            print(ecrt)
            print("Embeddings size: ", ecrt.shape)
            count_dict = {}

            print(ecrt.shape[0])
            for i in range(ecrt.shape[0]):

                if (ecrt[i] == ecrt[freq]).sum().item() == len(ecrt[i]):
                    count_dict[i] = 0
                if i % 10000 == 0:
                    print(i, len(count_dict))
                '''

                found = 0
                for j in count_dict:
                    if ecrt[i][0].item() != ecrt[j][0].item(): continue
                    if (ecrt[i] == ecrt[j]).sum().item() == len(ecrt[i]):
                        count_dict[j] += 1
                        found = 1
                        break
                if found == 0:
                    count_dict[i] = 1  # i is index of first encounter
                
                if i % 10000 == 0: 
                    cd = sorted(count_dict.items(),key=operator.itemgetter(1),reverse=True)
                    print(i, cd[:10])
                '''                

    return count_dict

def get_nodeId_Map(scenario):
    path_dir = "../data/E3/graph_data"
    filename_node_edge_types = "../examples/node_edge_types.pickle"
    nodeType_map, edgeType_map = load_node_edge_types(filename_node_edge_types)
    print("\nNode types:\n", nodeType_map)
    print("\nEdge types:\n", edgeType_map)

    _, nodeId_Map = index_dataset([scenario], path_dir, nodeType_map, edgeType_map)

    return nodeId_Map


def read_embeddings(filename_emb):

    emb_dict = {}
    uuid_dict = {}
    print("Reading embeddings from: ", filename_emb)
    f = open(filename_emb, 'r')
    count = 0
    for line in f:
        temp = line.strip('\n').split(' ')
        uuid_dict[count] = temp[0]
        emb_dict[count] = [float(x) for x in temp[1:]]
        count += 1
    f.close()
    return emb_dict, uuid_dict


def write_embeddings_pickle(emb_dict, fileout):
    emb_list = list(emb_dict.values())
    print(emb_list[:2])
    embeddings = torch.tensor(emb_list, dtype=torch.float)
    with open(fileout, 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Embeddings saved in file: ", fileout)


def convert_embeddings(filein, fileout):
    emb_dict, _, = read_embeddings(filein)
    write_embeddings_pickle(emb_dict, fileout)


def compare_emb(scenario, emb_list):

    h_feats = 256 # embeddings dim
    lr = 0.001  # learning rate
    epochs = 300
    seed_emb = 5

    print("\nUsing GraphSage Embeddings with Parameters:")

    param_str = 'dim' + str(h_feats) + '_seed' + str(seed_emb) + '_lr' + str(lr)[2:] + '_epochs' + str(epochs)
    print("Embeddings Size: ", h_feats)
    print("Embeddings Training Epochs: ", epochs)
    print("Embeddings Model (GNNs) Learning Rate: ", lr)

    path_emb = "../embeddings/train_on_trace_fivedirections_cadets/" + param_str
    #path_emb = "../embeddings/train_on_trace/" + param_str

    files_emb = {'cadets': os.path.join(path_emb, 'emb_cadets.pickle'),
                 'trace': os.path.join(path_emb, 'emb_trace.pickle'),
                 'theia': os.path.join(path_emb, 'emb_theia.pickle'),
                 'fivedirections': os.path.join(path_emb, 'emb_fivedirections.pickle')}

    filename_emb = files_emb[scenario]
    print("Loading embeddings from: ", filename_emb)
    with open(filename_emb, 'rb') as handle:
        ecrt = pickle.load(handle)

        print("\nEmbeddings:")
        print(ecrt)
        print("Embeddings size: ", ecrt.shape)

        print(ecrt.shape[0])
        for i in range(ecrt.shape[0]):
            if (str)(ecrt[i][0].item())[:5] != (str)(emb_list[i][0])[:5]: 
                print('found', i, ecrt[i][0].item(), emb_list[i][0])

            if i % 10000 == 0:
                print('line:', i)
        
def get_node_edge_types(scenario_list, path_dir):
    edgeType_map = {}  # type -> id
    nodeType_map = {}  # type -> id
    nodeType_cnt = 0
    edgeType_cnt = 0

    # save info by edge
    for scenario in scenario_list:
        path = os.path.join(path_dir, scenario, "{}_test.txt".format(scenario))
        print("processing file: ", path)
        f = open(path, 'r')

        for line in f:
            temp = line.strip('\n').split('\t')

            # convert node types to a numeric value
            if not (temp[1] in nodeType_map.keys()):
                nodeType_map[temp[1]] = nodeType_cnt
                nodeType_cnt += 1

            if not (temp[3] in nodeType_map.keys()):
                nodeType_map[temp[3]] = nodeType_cnt
                nodeType_cnt += 1

            # convert from edge type string to a numeric value
            if not (temp[4] in edgeType_map.keys()):
                edgeType_map[temp[4]] = edgeType_cnt
                edgeType_cnt += 1

    print("nodeType_map: ", nodeType_map)
    print("edgeType_map: ", edgeType_map)

    '''
    mappings_dict = {}
    mappings_dict['node'] = nodeType_map
    mappings_dict['edge'] = edgeType_map

    # save mappings to file
    filename_examples = 'node_edge_types.pickle'
    path_examples = '../examples'
    os.makedirs(path_examples, exist_ok = True)
    filename_examples = os.path.join(path_examples, filename_examples)

    with open(filename_examples, 'wb') as handle:
        pickle.dump(mappings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Node edge mappings have been saved in: ", filename_examples)

    '''

    return nodeType_map, edgeType_map


if __name__ == "__main__":
    
    scenario='fivedirections'

    path_dir = "../data/E3/graph_data"
    get_node_edge_types([scenario], path_dir)


    exit()


    #nodeId_map = get_nodeId_Map(scenario)
    #keys = list(nodeId_map.keys())

    #print(len(keys))
    #print(list(nodeId_map.items())[:10])

    path = "/net/data/idsgnn/darpatc/E3/embeddings/mcas-gmra/graphsage"
    filein = "{}/emb_{}_dram.txt".format(path, scenario)
    fileout = "{}/emb_{}_dram.pickle".format(path, scenario)
    convert_embeddings(filein, fileout)

    exit()

    filename_emb = "/net/data/idsgnn/darpatc/E3/embeddings/mcas-gmra/emb_{}_test_256.txt".format(scenario) 
    emb_dict, uuid_dict = read_embeddings(filename_emb)

    filename_emb2 = "/net/data/idsgnn/darpatc/E3/embeddings/mcas-gmra/emb_{}_dram.txt".format(scenario) 
    emb_dict2, uuid_dict2 = read_embeddings(filename_emb2)

    #compare_emb(scenario, emb_dict)
    for i in uuid_dict:
        if uuid_dict[i] != uuid_dict2[i]:
            print('diff:', i, uuid_dict[i], uuid_dict2[i])

    print('no diff')
    exit()


    uuids = dict()
    count_dict = get_freq_node()

    #print(len(count_dict))

    for i in count_dict:
        uuid = keys[i]
        uuids[uuid] = 0
        if i % 10000 == 0:
            print(i, len(uuids))

    #print(len(uuids))

    path = os.path.join("../data/E3/graph_data", scenario, scenario + '_test.txt') 

    print("processing file: ", path)
    f = open(path, 'r')
    
    srcs = set()
    types = set()
    events = set()

    i = 0
    for line in f:
        i = i + 1
        temp = line.strip('\n').split('\t')
        if temp[2] in uuids:
            srcs.add(temp[0])
            types.add(temp[3])
            events.add(temp[4])
        #if i % 10000 == 0:
        #    print(i)

    print(list(uuids.keys()))
    print(len(uuids), len(keys))
    print(srcs)
    print(types)
    print(events)

    #nid = 28
    #uuid = list(nodeId_map.keys())[list(nodeId_map.values()).index(nid)]
    #print(uuid)
    
