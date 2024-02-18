# https://docs.dgl.ai/en/0.8.x/tutorials/blitz/4_link_predict.html

import dgl
import os
import torch
import itertools
import numpy as np
import pandas as pd
import dgl.function as fn
from dgl.nn import SAGEConv
import random
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from supervised_examples import *


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            #print("Embeddings: ", h)
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            scores = g.edata['score'][:, 0]
            return scores


# https://docs.dgl.ai/en/0.7.x/tutorials/blitz/4_link_predict.html
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_logreg(X_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
    # print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    return logreg

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    #clf = RandomForestClassifier()
    #clf.fit(X_train, y_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    #print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    return clf


def link_scoring(embeddings, data_split):

    num_nodes = len(embeddings)

    # build new graphs that only contain the positive or the negative edges,
    # because we can then apply the API on these new graphs
    pos_u = data_split["pos"][0]
    pos_v = data_split["pos"][1]

    neg_u = data_split["neg"][0]
    neg_v = data_split["neg"][1]

    # construct the "positive graph" and the "negative graph" for training
    pos_g = dgl.graph((pos_u, pos_v), num_nodes=num_nodes)
    #print("positive graph:", pos_g)
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=num_nodes)
    #print("negative graph:", neg_g)

    # predict a score per link
    # this is a dot-product between the embeddings of each pair of nodes per link;
    # other predictor functions can be used too
    pred = DotPredictor()
    
    #emb_dim = embeddings.shape[1]
    #print('before pred:', emb_dim)
    #pred = MLPPredictor(emb_dim)

    pos_score = pred(pos_g, embeddings)
    neg_score = pred(neg_g, embeddings)
    
    return pos_score, neg_score


def main():

    seed = 5 # 1 and 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    h_feats = 256
    
    print("\nUsing Normalized GMRA Embeddings based on GraphSage")

    path_emb = "/net/data/idsgnn/darpatc/E3/embeddings/mcas-gmra/graphsage/normalized/"
    files_emb = {'cadets': os.path.join(path_emb, 'emb_cadets_dram.pickle'), 
                 'trace': os.path.join(path_emb, 'emb_trace_dram.pickle'), 
                 'theia': os.path.join(path_emb, 'emb_theia_dram.pickle'),
                 'fivedirections': os.path.join(path_emb, 'emb_fivedirections_dram.pickle')}
    '''

    print("\nUsing GraphSage Embeddings with Parameters:")

    lr = 0.001  # learning rate
    epochs = 300
    seed_emb = 5

    param_str = 'dim' + str(h_feats) + '_seed' + str(seed_emb) + '_lr' + str(lr)[2:] + '_epochs' + str(epochs)
    print("Embeddings Size: ", h_feats)
    print("Embeddings Training Epochs: ", epochs)
    print("Embeddings Model (GNNs) Learning Rate: ", lr)

    path_emb = "../embeddings/train_on_trace_fivedirections_cadets/" + param_str
    files_emb = {'cadets': os.path.join(path_emb, 'emb_cadets.pickle'),
                 'trace': os.path.join(path_emb, 'emb_trace.pickle'),
                 'theia': os.path.join(path_emb, 'emb_theia.pickle'),
                 'fivedirections': os.path.join(path_emb, 'emb_fivedirections.pickle')}
    '''

    # -----

    # scenario_train = ['trace', 'fivedirections', 'cadets'] # train on multiple
    scenario_train = ['trace'] # train on multiple
    scenario_test = 'trace' # generate embeddings for these
    # scenario_list = ['trace', 'fivedirections', 'cadets', 'theia'] #all

    print("\n--- Link Prediction --- ")
    print("\nTrain Datasets: ", scenario_train)
    print("Test Dataset: ", scenario_test)

    mypath_dir = "../data/E3/graph_data"
    filename_examples = "examples_uuid.pickle"
    
    # get all the types of nodes and edges in the system
    filename_node_edge_types = "../examples/node_edge_types.pickle"
    nodeType_map, edgeType_map = load_node_edge_types(filename_node_edge_types)
    print("\nNode types:\n", nodeType_map)
    print("\nEdge types:\n", edgeType_map)

    # --- process train data to get link scores based on embeddings

    print("\n--- Processing TRAIN Data")

    # read node embeddings for the train data sets
    print("")
    embeddings_train = torch.empty((0, h_feats), dtype=torch.float)
    for scenario in scenario_train: 
        filename_emb = files_emb[scenario]
        print("Loading embeddings from: ", filename_emb)
        with open(filename_emb, 'rb') as handle:
            ecrt = pickle.load(handle)
            embeddings_train = torch.cat([embeddings_train, ecrt])
    print("\nEmbeddings:")
    print(embeddings_train)
    print("Embeddings size: ", embeddings_train.shape[1])

    # train data: construct positive and negative examples for links
    # construct_examples_supervised(scenario_train, mypath_dir, nodeType_map, edgeType_map)

    print("\nTraining samples represent links with malicious source node from: ", scenario_train)
    
    data_split_train, _, _, _ = load_examples_supervised(scenario_train, filename_examples)

    print("{} positive examples (actual links)".format(len(data_split_train["pos"][0])))
    print("{} negative examples (non-existent links)".format(len(data_split_train["neg"][0])))
    
    # train data: compute link scores
    print("\nComputing Link Scores as vector product of node embeddings:")
    pos_train, neg_train = link_scoring(embeddings_train, data_split_train)
    print("Scores for positive links: ", pos_train)
    print("Scores for negative links: ", neg_train)

    # --- process test data to get link scores based on embeddings

    print("\n--- Processing TEST Data")

    filename_emb = files_emb[scenario_test]
    print("\nLoading embeddings from: ", filename_emb)
    with open(filename_emb, 'rb') as handle:
        embeddings_test = pickle.load(handle)
    print("\nEmbeddings:")
    print(embeddings_test)
    print("Embeddings size: ", embeddings_test.shape[1])

    # test data: construct positive and negative examples for links
    # construct_examples_supervised([scenario_test], mypath_dir, nodeType_map, edgeType_map, posSrc=True, negSrc=True)
    print("\nTest samples represent links with malicious source node from: ", scenario_test)
    
    pos_neg_test_index = 18

    #data_split_test, pos_src_test, neg_src_test, data_split_pos_neg = load_examples_supervised([scenario_test], filename_examples, posSrc=True, negSrc=True, index=pos_neg_test_index)
    data_split_test, pos_src_test, neg_src_test, data_split_pos_neg = load_examples_supervised([scenario_test], filename_examples)
    print("{} positive examples (actual links)".format(len(data_split_test["pos"][0])))
    print("{} negative examples (non-existent links)".format(len(data_split_test["neg"][1])))
   
    #print("pos_src_test: ", len(pos_src_test))
    #print("neg_src_test: ", len(neg_src_test))
    #print(data_split_pos_neg)

    # test data: compute test scores
    print("\nComputing Link Scores as vector product of node embeddings:")
    pos_test, neg_test = link_scoring(embeddings_test, data_split_test)
    print("Scores for positive links: ", pos_test)
    print("Scores for negative links: ", neg_test)

    # run the supervised classification task, which uses ground truth (malicious nodes)
    # train a classifier on the pos/neg examples from training
    # test prediction on pos/neg test examples
    print("\n--- Link Classification")
    
    with torch.no_grad():

        print("\nTraining the classifier on the following data:")
        X_train = torch.cat([pos_train, neg_train]).numpy().reshape(-1, 1)
        y_train = torch.cat([torch.ones(pos_train.shape[0]), torch.zeros(neg_train.shape[0])]).numpy()
        print("X_train (link scores):\n", X_train)
        print("y_train (link labels 1/0):" , y_train)

        #print("\nTrain on: ", scenario_train, " Test on: ", scenario_test)
        
        logreg = train_logreg(X_train, y_train)

        clf = train_decision_tree(X_train, y_train)

        print("Training finished")

        print("\nApplying the trained classifier on the following data:")
        
        X_test = torch.cat([pos_test, neg_test]).numpy().reshape(-1, 1)
        y_test = torch.cat([torch.ones(pos_test.shape[0]), torch.zeros(neg_test.shape[0])]).numpy()

        print("X_test (link scores):\n", X_test)
        
        print("\nTest labels for evaluation of prediction accuracy:")
        print("y_test (link labels 1/0):" , y_test)
        
        logreg_acc = logreg.score(X_test, y_test)

        clf_acc = clf.score(X_test, y_test)

        print("\nAccurracy of Logistic Regression: {:.2f}".format(logreg_acc))    
        print("Accurracy of Decision Tree: {:.2f}".format(clf_acc)) 
        print("\n")

        '''
        logreg_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
        logreg_pr_auc = average_precision_score(y_test, logreg.predict(X_test))
        print("Logistic Regression -- ROC AUC: {}, PR AUC: {}".format(logreg_roc_auc, logreg_pr_auc))

        clf_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
        clf_pr_auc = average_precision_score(y_test, clf.predict(X_test))
        print("Decision Tree -- ROC AUC: {}, PR AUC: {}".format(clf_roc_auc, clf_pr_auc))
        '''

        '''
        print("\n--- Testing on a specific alert")

        #print(list(pos_src_test.items())[pos_neg_test_index])
        #print(list(neg_src_test.items())[pos_neg_test_index])

        mal_uuid = list(pos_src_test.keys())[pos_neg_test_index]
        print("\nMalicious uuid: ", mal_uuid)

        print("\nComputing Link Scores as vector product of node embeddings")
        pos_test, neg_test = link_scoring(embeddings_test, data_split_pos_neg)
        print("Scores for positive links: ", pos_test)
        print("Scores for negative links: ", neg_test)

        print("\nApplying the trained classifier")

        X_test = torch.cat([pos_test, neg_test]).numpy().reshape(-1, 1)
        y_test = torch.cat([torch.ones(pos_test.shape[0]), torch.zeros(neg_test.shape[0])]).numpy()

        #print("X_test (link scores):\n", X_test)

        #print("\nTest labels for evaluation of prediction accuracy:")
        #print("y_test (link labels 1/0):" , y_test)

        logreg_acc = logreg.score(X_test, y_test)
        clf_acc = clf.score(X_test, y_test)

        print("\nAccurracy of Logistic Regression: {:.2f}".format(logreg_acc))
        print("Accurracy of Decision Tree: {:.2f}".format(clf_acc))
        print("")

        lrprob = clf.predict_proba(X_test)[:,1]
        print("Probabilities of links being malicious: ", lrprob)

        zipped = list(zip(list(lrprob), list(y_test)))
        res = sorted(zipped, key = lambda x: x[0], reverse=True)
        print("\nTop ten links for malicious source {}:".format(mal_uuid))
        for i in res[:10]:
            print(i[0], i[1])
        print("")
        '''

if __name__ == "__main__":
    main()


