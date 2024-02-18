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
from sklearn.neighbors import KNeighborsClassifier

from data_process import *
from supervised_examples import *


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            #print("Embeddings: ", h, "size: ", h.shape)
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            scores = g.edata['score'][:, 0]
            return scores


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
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    #print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    return clf

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
    #print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))


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

    pos_score = pred(pos_g, embeddings)
    neg_score = pred(neg_g, embeddings)

    return pos_score, neg_score



def read_embeddings(filename_emb, emb_list, nodeId_map):

    print("Reading embeddings from: ", filename_emb)
    f = open(filename_emb, 'r')
    for line in f:
        temp = line.strip('\n').split(' ')
        if len(temp) == 2: 
            assert(int(temp[0]) <= len(nodeId_map))
        else:
            node_id = nodeId_map[temp[0]]
            emb_list[node_id] = [float(x) for x in temp[1:]]

    f.close()
    return emb_list


def main():

    seed = 5 # 1 and 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dim = 256 # embeddings dim
    
    '''
    print("\nUsing Node2Vec Embeddings with dim: {}".format(dim))

    path_emb = "../embeddings/node2vec/dim256"

    files_emb = {'cadets': os.path.join(path_emb, 'cadets_test_{}.txt'.format(dim)), 
                 'trace': os.path.join(path_emb, 'trace_test_{}.txt'.format(dim)), 
                 'theia': os.path.join(path_emb, 'theia_test_{}.txt'.format(dim)), 
                 'fivedirections': os.path.join(path_emb, 'fivedirections_test_{}.txt'.format(dim)) 
                 }
    '''

    print("\nUsing GMRA + Node2Vec Embeddings with dim: {}".format(dim))

    path_emb = "/net/data/idsgnn/darpatc/E3/embeddings/mcas-gmra/node2vec"

    files_emb = {'cadets': os.path.join(path_emb, 'cadets_test_{}_{}.txt'.format(dim, 'dram')),
                 'trace': os.path.join(path_emb, 'trace_test_{}_{}.txt'.format(dim, 'dram')),
                 'theia': os.path.join(path_emb, 'theia_test_{}_{}.txt'.format(dim, 'dram')),
                 'fivedirections': os.path.join(path_emb, 'fivedirections_test_{}_{}.txt'.format(dim, 'dram'))
                 }

    #scenario_train = ['trace', 'fivedirections', 'cadets'] # train on multiple
    scenario_train = ['theia'] # train on multiple
    #scenario_train = ['trace'] # train on multiple
    scenario_test = 'cadets' 
    mypath_dir = "../data/E3/graph_data"
    filename_examples = "examples_uuid.pickle"

    # get all the types of nodes and edges in the system
    
    # scenario_list = ['trace', 'fivedirections', 'cadets', 'theia'] #all
    # nodeType_map, edgeType_map = get_node_edge_types(scenario_list, mypath_dir)
    filename_node_edge_types = "../examples/node_edge_types.pickle"
    nodeType_map, edgeType_map = load_node_edge_types(filename_node_edge_types)

    # --- process train data to get link scores based on embeddings

    # train data: index uuids to ids
    provenance_train, nodeId_map_train = index_dataset(scenario_train, mypath_dir, nodeType_map, edgeType_map)

    # read node embeddings for the train data sets
    emb_list = [None] * len(nodeId_map_train)
    for scenario in scenario_train: 
        filename_emb = files_emb[scenario]
        emb_list = read_embeddings(filename_emb, emb_list, nodeId_map_train)
    embeddings_train = torch.tensor(emb_list, dtype=torch.float)
    print("\nEmbeddings:")
    print(embeddings_train)
    print("Embeddings size: ", embeddings_train.shape[1])

    # train data: construct positive and negative examples for links
    # construct_examples_supervised(scenario_train, mypath_dir, nodeType_map, edgeType_map)
    data_split_train, _, _, _ = load_examples_supervised(scenario_train, filename_examples)

    print("\nTraining samples represent links with malicious source node from: ", scenario_train)
    print("{} positive examples (actual links)".format(len(data_split_train["pos"][0])))
    print("{} negative examples (non-existent links)".format(len(data_split_train["neg"][0])))

    # train data: compute link scores
    print("\nComputing Link Scores as vector product of node embeddings:")
    pos_train, neg_train = link_scoring(embeddings_train, data_split_train)
    print("Scores for positive links: ", pos_train)
    print("Scores for negative links: ", neg_train)


    # --- process test data to get link scores based on embeddings

    # test data: index uuids to ids
    provenance_test, nodeId_map_test = index_dataset([scenario_test], mypath_dir, nodeType_map, edgeType_map)
    
    # read node embeddings for the test data set
    filename_emb = files_emb[scenario_test]
    emb_list = [None] * len(nodeId_map_test)
    emb_list = read_embeddings(filename_emb, emb_list, nodeId_map_test)
    embeddings_test = torch.tensor(emb_list, dtype=torch.float)

    # test data: construct positive and negative examples for links
    # construct_examples_supervised([scenario_test], mypath_dir, nodeType_map, edgeType_map, posSrc=True, negSrc=True)
    print("\nTest samples represent links with malicious source node from: ", scenario_test)

    data_split_test, pos_src_test, neg_src_test, data_split_pos_neg = load_examples_supervised([scenario_test], filename_examples)
    print("{} positive examples (actual links)".format(len(data_split_test["pos"][0])))
    print("{} negative examples (non-existent links)".format(len(data_split_test["neg"][1])))

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
        #knn = train_knn(X_train, y_train)

        print("Training finished")

        print("\nApplying the trained classifier on the following data:")

        X_test = torch.cat([pos_test, neg_test]).numpy().reshape(-1, 1)
        y_test = torch.cat([torch.ones(pos_test.shape[0]), torch.zeros(neg_test.shape[0])]).numpy()

        print("X_test (link scores):\n", X_test)

        print("\nGround truth Labels used to evaluate prediction accuracy:")
        print("y_test (link labels 1/0):" , y_test)

        logreg_acc = logreg.score(X_test, y_test)

        clf_acc = clf.score(X_test, y_test)

        #knn_acc = knn.score(X_test, y_test)

        print("\nAccurracy of Logistic Regression: {:.2f}".format(logreg_acc))
        print("Accurracy of Decision Tree: {:.2f}".format(clf_acc))
        print("\n")

        '''
        print("Accurracies Logistic Regression={}, Decision Tree={}".format(logreg_acc, clf_acc))    

        clf_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
        clf_pr_auc = average_precision_score(y_test, clf.predict(X_test))
        print("Decision Tree -- ROC AUC: {}, PR AUC: {}".format(clf_roc_auc, clf_pr_auc))

        logreg_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
        logreg_pr_auc = average_precision_score(y_test, logreg.predict(X_test))
        print("Logistic Regression -- ROC AUC: {}, PR AUC: {}".format(logreg_roc_auc, logreg_pr_auc))
        '''


if __name__ == "__main__":
    main()


