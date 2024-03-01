# based on Threatrace code

import os
import pickle
import torch
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import dgl
from dgl.data import DGLDataset


# constructing the dataset
class DarpaTCDataset(DGLDataset):

    graph_info = {}
    scaler = MinMaxScaler()
    training = True
    use_scaler = True

    def __init__(self, graph_info, scaler, training=True, use_scaler=True):
        self.graph_info = graph_info
        self.scaler = scaler
        self.training = training
        self.use_scaler = use_scaler
        super().__init__(name='darpa_tc')

    def process(self):
       
        x_list = self.graph_info['x_list']
        y_list = self.graph_info['y_list']
        edge_types = self.graph_info['edge_types']
        edge_s = self.graph_info['edge_s']
        edge_d = self.graph_info['edge_d']

        # x_list contains count-based node features and centrality-based node features
        # normalize feature values to the default [0, 1] range

        if self.use_scaler:
            if self.training:
                self.scaler.fit(x_list)
            x_list = self.scaler.transform(x_list)

        node_features = torch.tensor(x_list, dtype=torch.float)
        node_labels = torch.tensor(y_list, dtype=torch.long)
        
        # torch, numeric
        edge_s = np.array(edge_s)
        edge_d = np.array(edge_d)
        edges_src = torch.from_numpy(edge_s)
        edges_dst = torch.from_numpy(edge_d)
        #edge_types = np.array(edge_types)
        #edge_features = torch.from_numpy(edge_types)

        #self.graph = dgl.graph((edges_src, edges_dst), num_nodes=y_list.shape[0])
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(y_list))
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        # self.graph.edata['type'] = edge_features
        print("features: ", node_features)


    def __getitem__(self, i):
        return self.graph, self.scaler

    def __len__(self):
        return 1


def centrality_features(node_cnt, nodeId_map, scenario_list, centralities, 
        feature_path="../data/E3/graph_data/features", subdir="test"):

    features_num = len(centralities)
    if features_num == 0:
        return []

    # node_cnt and node_index are across all scenarios involved in training and testing

    # initialize features
    x_list = []
    for i in range(node_cnt):
        temp_list = []
        for j in range(features_num):
            temp_list.append(0)
        x_list.append(temp_list)

    for scenario in scenario_list:
        filepath = os.path.join(feature_path, scenario, subdir)
        feature_index = 0

        for feature_name in centralities:
            filename = os.path.join(filepath, feature_name + ".pickle")
            
            # load the feature dictionary
            with open(filename, 'rb') as handle:
                feature_dict = pickle.load(handle)

            for uuid in feature_dict:
                node_index = nodeId_map[scenario][uuid]
                x_list[node_index][feature_index] = feature_dict[uuid]
            
            feature_index += 1

    return x_list


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

    mappings_dict = {}
    mappings_dict['node'] = nodeType_map
    mappings_dict['edge'] = edgeType_map

    # save mappings to file
    filename_examples = 'node_edge_types.pickle'
    path_examples = 'link_prediction/examples'
    os.makedirs(path_examples, exist_ok = True)
    filename_examples = os.path.join(path_examples, filename_examples)

    with open(filename_examples, 'wb') as handle:
        pickle.dump(mappings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Node edge mappings have been saved in: ", filename_examples)

    return nodeType_map, edgeType_map


def load_node_edge_types(filename_examples):
    with open(filename_examples, 'rb') as handle:
        mappings_dict = pickle.load(handle)

    return mappings_dict['node'], mappings_dict['edge']


def index_dataset(scenario_list, path_dir, nodeType_map, edgeType_map):
    
    node_cnt = 0
    
    provenance = []
    nodeId_map = {}  # scenario: uuid -> id
    
    # save info by edge
    print(scenario_list)
    print(path_dir)

    for scenario in scenario_list:
        print(scenario)
        path = os.path.join(path_dir, scenario, "{}_test.txt".format(scenario))
        print("processing file: ", path)
        f = open(path, 'r')

        for line in f:
            temp = line.strip('\n').split('\t')

            # convert uuid to id
            if not (temp[0] in nodeId_map.keys()):
                nodeId_map[temp[0]] = node_cnt
                node_cnt += 1
            temp[0] = nodeId_map[temp[0]]    

            if not (temp[2] in nodeId_map.keys()):
                nodeId_map[temp[2]] = node_cnt
                node_cnt += 1
            temp[2] = nodeId_map[temp[2]]

            # convert node types to a numeric value
            temp[1] = nodeType_map[temp[1]]
            temp[3] = nodeType_map[temp[3]]

            # convert from edge type string to a numeric value
            temp[4] = edgeType_map[temp[4]]
        
            # save the info for this edge
            provenance.append(temp)

    '''
    # save the node id map scenario: uuid -> id
    dir_nodeId = "nodeId_map"
    filename_nodeId = "nodeId_map"
    for scenario in scenario_list:
        filename_nodeId += '_' + scenario
    filename_nodeId += '.pickle'
    filename_nodeId = os.path.join(dir_nodeId, filename_nodeId)
    with open(filename_nodeId, 'wb') as handle:
        pickle.dump(nodeId_map, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    '''

    return provenance, nodeId_map 



def compute_features(provenance, edgeType_map, nodeId_map, scenario_list,
        edge_count_features, centralities, feature_path="../data/E3/graph_data/features", subdir="test"):

    graph_info = {}
    edge_s = []
    edge_d = []
    edge_types = []

    feature_num = len(edgeType_map)
    node_cnt = len(nodeId_map)
        
    # initialize features
    x_list = []
    y_list = []
    for i in range(node_cnt):
        temp_list = []
        for j in range(feature_num * 2):
            temp_list.append(0)
        x_list.append(temp_list)
        y_list.append(0)
    
    # constructing the node features as counts per edge type
    for temp in provenance:
        # print(temp)
        srcId = temp[0]
        srcType = temp[1]
        dstId = temp[2]
        dstType = temp[3]
        edge = temp[4]
        
        x_list[srcId][edge] += 1 # outgoing edge
        y_list[srcId] = srcType
        x_list[dstId][edge+feature_num] += 1  # incoming edge
        y_list[dstId] = dstType

        # topology info, node ids that form this edge
        edge_s.append(temp[0])
        edge_d.append(temp[2])
        edge_types.append(edge)
    
    # add centrality features per node
    # the number of features will be feature_num * 2 + len(centralities)
    x_list_centralities = centrality_features(node_cnt, nodeId_map, scenario_list,
        centralities=centralities,
        feature_path=feature_path, subdir=subdir)

    if not edge_count_features:
        x_list = x_list_centralities

    if len(centralities) > 0:
        for i in range(node_cnt):
            x_list[i] = x_list[i] + x_list_centralities[i]

    graph_info['x_list'] = x_list
    graph_info['y_list'] = y_list
    graph_info['edge_types'] = edge_types
    graph_info['edge_s'] = edge_s
    graph_info['edge_d'] = edge_d

    return graph_info

    #return x_list, y_list, edge_types, edge_s, edge_d


