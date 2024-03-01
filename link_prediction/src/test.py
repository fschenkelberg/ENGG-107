import dgl
import os
import torch
import numpy as np
import dgl.function as fn
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import pymc3 as pm

from supervised_examples import *

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            scores = g.edata['score'][:, 0]
            return scores

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def train_bayesian_regression(X_train, y_train):
    # Create a PyMC3 model
    with pm.Model() as model:
        # Priors for the parameters
        slope = pm.Normal('slope', mu=0, sd=10)
        intercept = pm.Normal('intercept', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of the outcome
        mu = intercept + slope * X_train

        # Likelihood (sampling distribution) of the observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y_train)

        # Run the MCMC sampling
        trace = pm.sample(2000, tune=1000)
        
    return trace


def link_scoring(embeddings, data_split):
    num_nodes = len(embeddings)
    pos_u = data_split["pos"][0]
    pos_v = data_split["pos"][1]
    neg_u = data_split["neg"][0]
    neg_v = data_split["neg"][1]
    pos_g = dgl.graph((pos_u, pos_v), num_nodes=num_nodes)
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=num_nodes)
    pred = DotPredictor()
    pos_score = pred(pos_g, embeddings)
    neg_score = pred(neg_g, embeddings)
    return pos_score, neg_score

def train_classifier(X_train, y_train, classifier):
    return classifier.fit(X_train, y_train)

def main():
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    h_feats = 32
    
    scenario_train = ['trace']
    scenario_test = 'trace'

    path_emb = "/thayerfs/home/f006dg0/Malicious/32/"
    files_emb = {'cadets': os.path.join(path_emb, 'emb_cadets_test_32.pickle'), 
                 'trace': os.path.join(path_emb, 'emb_trace_test_32.pickle'), 
                 'theia': os.path.join(path_emb, 'emb_theia_test_32.pickle'),
                 'fivedirections': os.path.join(path_emb, 'emb_fivedirections_test_32.pickle')}

    filename_examples = "examples_uuid.pickle"
    
    # Load embeddings for training
    embeddings_train = torch.empty((0, h_feats), dtype=torch.float)
    for scenario in scenario_train: 
        filename_emb = files_emb[scenario]
        with open(filename_emb, 'rb') as handle:
            ecrt = pickle.load(handle)
            embeddings_train = torch.cat([embeddings_train, ecrt])

    # Load data split for training
    data_split_train, _, _, _ = load_examples_supervised(scenario_train, filename_examples)

    # Compute link scores for training data
    pos_train, neg_train = link_scoring(embeddings_train, data_split_train)

    # Load embeddings for testing
    filename_emb_test = files_emb[scenario_test]
    with open(filename_emb_test, 'rb') as handle:
        embeddings_test = pickle.load(handle)

    # Load data split for testing
    data_split_test, _, _, _ = load_examples_supervised([scenario_test], filename_examples)

    # Compute link scores for testing data
    pos_test, neg_test = link_scoring(embeddings_test, data_split_test)

    with torch.no_grad():
        X_train = torch.cat([pos_train, neg_train]).numpy().reshape(-1, 1)
        y_train = torch.cat([torch.ones(pos_train.shape[0]), torch.zeros(neg_train.shape[0])]).numpy()
        
        # Parallelize training of classifiers
        classifiers = Parallel(n_jobs=-1)(delayed(train_classifier)(X_train, y_train, classifier) 
                                          for classifier in [LogisticRegression()])

        X_test = torch.cat([pos_test, neg_test]).numpy().reshape(-1, 1)
        y_test = torch.cat([torch.ones(pos_test.shape[0]), torch.zeros(neg_test.shape[0])]).numpy()

        # Evaluate classifiers
        logreg_acc = classifiers[0].score(X_test, y_test)

        print("\nAccuracy of Logistic Regression: {:.2f}".format(logreg_acc))
        print("\n")

if __name__ == "__main__":
    main()
