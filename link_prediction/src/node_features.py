import pandas as pd
import networkx as nx
import numpy as np
import os
import pickle

#assumes tab delimiter
def get_nx_graph(filename):
    df = pd.read_csv(filename, delimiter="\t", names=["node1", "node1_attr", "node2", "node2_attr", "event_type", "timestamp"])
    G=nx.from_pandas_edgelist(df, 'node1', 'node2', ['event_type', 'timestamp'], nx.MultiDiGraph())
    node1_attr = df[['node1', 'node1_attr']].drop_duplicates().set_index('node1').to_dict('index')
    node2_attr = df[['node2', 'node2_attr']].drop_duplicates().set_index('node2').to_dict('index')
    nx.set_node_attributes(G, node1_attr)
    nx.set_node_attributes(G, node2_attr)
    return G, df


def get_overlap_stats(G1,G2):
    print("====OVERLAP STATS====")
    #How large is the intersection?
    I = nx.intersection(G1, G2)
    print(f"The two networks share {I.number_of_nodes()} nodes and {I.number_of_edges()} edges")
    print()


# focusing on the test files from each scenario, because the train files do not have attacks
def save_feature(feature_dict, pathdir, scenario, feature_name, subdir="test"):
    filepath = os.path.join(pathdir, scenario, subdir)
    os.makedirs(filepath, exist_ok = True)

    filename = os.path.join(filepath, feature_name + ".pickle")
    with open(filename, 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved to file: ", filename)


def main():
    mypath_dir = "../data/E3/graph_data"
    scenario = "trace"

    filename = os.path.join(mypath_dir, scenario, scenario + "_test.txt")
    print("processing file: ", filename)

    #feature_path = os.path.join(mypath_dir, "features") 

    # build the graph
    G1, df1 = get_nx_graph(filename)
    print(G1)

    # computing and saving degree centrality per node
    feature_name = "degree_centrality"
    degreec = nx.degree_centrality(G1)
    save_feature(degreec, feature_path, scenario, feature_name, subdir="test")

    # computing and saving in-degree centrality per node
    feature_name = "in_degree_centrality"
    in_degreec = nx.in_degree_centrality(G1)
    save_feature(in_degreec, feature_path, scenario, feature_name, subdir="test")

    # computing and saving out-degree centrality per node
    feature_name = "out_degree_centrality"
    out_degreec = nx.out_degree_centrality(G1)
    save_feature(out_degreec, feature_path, scenario, feature_name, subdir="test")

    # computing and saving pagerank per node
    feature_name = "pagerank"
    pr = nx.pagerank(G1)
    save_feature(pr, feature_path, scenario, feature_name, subdir="test")

    # computing and saving load centrality per node
    feature_name = "load_centrality"
    loadc = nx.load_centrality(G1)
    save_feature(loadc, feature_path, scenario, feature_name, subdir="test")
    
    # computing and saving closeness centrality per node
    feature_name = "closeness_centrality"
    closec = nx.closeness_centrality(G1)
    save_feature(closec, feature_path, scenario, feature_name, subdir="test")



if __name__ == "__main__":
    main()




