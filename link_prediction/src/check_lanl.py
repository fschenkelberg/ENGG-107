import pandas as pd
import networkx as nx
import numpy as np
import os


# 1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C1250,C586,NTLM,Network,LogOn,Success

#assumes tab delimiter
def get_nx_graph(filename):
    df = pd.read_csv(filename, delimiter=",", names=["ts", "user_src", "user_dst", "computer_src", "computer_dst", "authentication_type", "logon_type", "authentication_orientation", "success_fail_status"])
    G = nx.from_pandas_edgelist(df, 'user_src', 'user_dst', ["authentication_type", "logon_type", "authentication_orientation"], nx.MultiDiGraph())
    G1 = nx.from_pandas_edgelist(df, 'computer_src', 'computer_dst', ["authentication_type", "logon_type", "authentication_orientation"], nx.MultiDiGraph())

    print(G)
    print(G1)

def get_key(row):
    return (row["ts"], row["user_src"], row["computer_src"], row["computer_dst"])

def main():
    df_set = set()
    red_set = set()
    #filename = "/net/data/idsgnn/lanl_red/auth_10mil_4.txt"  # 715 9273599 14 9274300
    #filename = "/net/data/idsgnn/lanl_red/auth_10mil_3.txt"  # 715 9283691 7 9284399
    filename = "/net/data/idsgnn/lanl_red/auth_10mil_8.txt"   # 715 9250254 21 9250948 

    filename_red = "/net/data/idsgnn/lanl_red/redteam.txt"
    #get_nx_graph(filename)

    df = pd.read_csv(filename, delimiter=",", names=["ts", "user_src", "user_dst", "computer_src", "computer_dst", "authentication_type", "logon_type", "authentication_orientation", "success_fail_status"])
    df_red = pd.read_csv(filename_red, delimiter=",", names=["ts", "user_src", "computer_src", "computer_dst"])

    '''
    for index, row in df_red.iterrows():
        mytuple = (row["ts"], row["user_src"], row["computer_src"], row["computer_dst"])
        red_set.add(mytuple)
    '''

    df_red['key'] = df_red.apply(get_key, axis=1)
    red_set = set(df_red['key'].unique())

    print("red_set done")

    '''
    for index, row in df.iterrows():
        mytuple = (row["ts"], row["user_src"], row["computer_src"], row["computer_dst"])
        df_set.add(mytuple)
    '''

    df['key'] = df.apply(get_key, axis=1)
    df_set = set(df['key'].unique())

    print("df_set done")

    intersection = red_set & df_set
    union = red_set | df_set

    print(len(red_set), len(df_set), len(intersection), len(union))



if __name__ == "__main__":
    main()


