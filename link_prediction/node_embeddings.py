
import dgl
import os
import torch
import itertools
import numpy as np
import pandas as pd
import dgl.function as fn
from dgl.nn import SAGEConv
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from data_process import *
from unsupervised_examples import *


# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        # g.ndata['h'] = h
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        #print("Embeddings: ", h)
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            scores = g.edata['score'][:, 0]

            return scores

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    #print("Scores: {} Labels: {}".format(scores, labels))
    
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train(graph_info, data_split, h_feats, epochs, lr): 
    
    # constructing the dataset
    # normalize features to [0, 1] range with MinMaxScaler
    scaler = MinMaxScaler()
    dataset = DarpaTCDataset(graph_info, scaler, training=True, use_scaler=True)

    # the graph
    g = dataset[0][0]
    scaler = dataset[0][1]
    print(g)

    # build new graphs that only contain the positive or the negative edges,
    # because we can then apply the API on these new graphs
    pos_u = data_split["pos"][0]
    pos_v = data_split["pos"][1]
    
    neg_u = data_split["neg"][0]
    neg_v = data_split["neg"][1]

    # construct the "positive graph" and the "negative graph" for training
    pos_g = dgl.graph((pos_u, pos_v), num_nodes=g.number_of_nodes())
    print("positive graph:", pos_g)
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
    print("negative graph:", neg_g)


    # node features used in graphsage
    features = g.ndata['feat']
    #print(features)
    
    in_feats = g.ndata['feat'].shape[1]
    print("input features, number of nodes: ", in_feats, g.number_of_nodes())

    # train the model 
    model = GraphSAGE(in_feats, h_feats)
    
    # predict a score per link
    # this is a dot-product between the embeddings of each pair of nodes per link;
    # other predictor functions can be used too
    pred = DotPredictor()

    # optimizing the model to minimize loss
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=lr)
    for e in range(epochs):
        
        # forward 
        h = model(g, features)
        #print(h)
        
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        #print("Pos: ", pos_score)
        #print("Neg: ", neg_score)

        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            roc_auc, pr_auc = compute_auc(pos_score, neg_score)
            print('In epoch {}, train loss: {}, train ROC: {}, train PR: {}'.format(e, loss, roc_auc, pr_auc))
    
    return model, scaler


def predict_embeddings(graph_info, model, scaler):

    # constructing the features dataset
    # normalize features to [0, 1] range with MinMaxScaler
    dataset = DarpaTCDataset(graph_info, scaler, False)

    # the graph
    g = dataset[0][0]
    print(g)

    # node features used in graphsage
    features = g.ndata['feat']
    #print(features)

    in_feats = g.ndata['feat'].shape[1]
    print("input features, number of nodes: ", in_feats, g.number_of_nodes())

    h = model(g, features)

    return h


def main():

    seed = 5 # 1 and 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    h_feats = 32 # embedding size
    epochs = 300
    lr = 0.001  # learning rate
 
    print("Parameters: embeddings_dim={}, lr={}, epochs={}, seed={}".format(h_feats, lr, epochs, seed))

    #scenario_train = ['trace'] 
    scenario_train = ['trace', 'fivedirections', 'cadets'] # can train on multiple
    scenario_test = ['trace', 'fivedirections', 'cadets', 'theia'] # generate embeddings for these
    mypath_dir = "../data/E3/graph_data"
   
    aux_dir = "train_on_trace_fivedirections_cadets"
    #aux_dir = "train_on_trace"
    param_str = 'dim' + str(h_feats) + '_seed' + str(seed) + '_lr' + str(lr)[2:] + '_epochs' + str(epochs)

    path_model = os.path.join('../model', aux_dir, param_str)
    os.makedirs(path_model, exist_ok=True)

    path_embeddings = os.path.join('../embeddings_dec1', aux_dir, param_str)
    os.makedirs(path_embeddings, exist_ok=True)
    
    # get all the types of nodes and edges in the system
    scenario_list = ['trace', 'fivedirections', 'cadets', 'theia'] #all
    nodeType_map, edgeType_map = get_node_edge_types(scenario_list, mypath_dir)
    
    # ---- train the embeddings model --------

    # index uuids to ids
    provenance_train, nodeId_map_train = index_dataset(scenario_train, mypath_dir, nodeType_map, edgeType_map)

    # construct positive and negative examples for links
    data_split = construct_examples_unsupervised(scenario_train, nodeId_map_train)
  
    # features will be edge counts, centralities do not seem to help
    centralities = []
    edge_count_features = True
    graph_info = compute_features(provenance_train, edgeType_map, nodeId_map_train, scenario_train, edge_count_features, centralities)
            
    # train model to learn how to compute node embeddings from context
    mymodel, myscaler = train(graph_info, data_split, h_feats, epochs, lr)

    # save the model; the scaler maps the feature values between 0 and 1
    model_and_scaler = {}
    model_and_scaler['model'] = mymodel
    model_and_scaler['scaler'] = myscaler
    filename_model = os.path.join(path_model, 'model.pickle')
    with open(filename_model, 'wb') as handle:
        pickle.dump(model_and_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nModel for learning embeddings saved in: ", filename_model)


    # ------ generate embeddings on any graph  -------

    # apply model to generate embeddings
    print("\nGenerating embeddings..")

    # get the trained embeddings model
    filename_model = os.path.join(path_model, 'model.pickle')
    with open(filename_model, 'rb') as handle:
        model_and_scaler = pickle.load(handle)
  
    mymodel = model_and_scaler['model']
    myscaler = model_and_scaler['scaler']

    for myscenario in scenario_test:

        # index uuids to ids
        provenance_test, nodeId_map_test = index_dataset([myscenario], mypath_dir, nodeType_map, edgeType_map)

        # compute features for the test dataset
        graph_info = compute_features(provenance_test, edgeType_map, nodeId_map_test, [myscenario], edge_count_features, centralities)
        embeddings = predict_embeddings(graph_info, mymodel, myscaler)

        print("\nEmbeddings: ", embeddings)
        
        # save embeddings to file
        filename_emb = os.path.join(path_embeddings, 'emb_{}.pickle'.format(myscenario))
        with open(filename_emb, 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Embeddings by id saved in file: ", filename_emb)
        
        key_list = list(nodeId_map_test.keys())
        val_list = list(nodeId_map_test.values())
        emb_dict = {}

        for i in range(len(embeddings)):
 
            # print key with val i
            position = val_list.index(i)
            uuid = key_list[position]
            emb_dict[uuid] = embeddings[i]
        
        # save embeddings to file
        filename_emb = os.path.join(path_embeddings, 'emb_uuid_{}.pickle'.format(myscenario))
        with open(filename_emb, 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Embeddings by uuid saved in file: ", filename_emb)
        

if __name__ == "__main__":
    main()


