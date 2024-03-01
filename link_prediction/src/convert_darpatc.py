import pickle
import os
import random
import numpy as np
import torch


def read_embeddings(filename_emb, expected_dim=32):
    emb_dict = {}
    uuid_dict = {}
    print("Reading embeddings from: ", filename_emb)
    f = open(filename_emb, 'r')
    count = 0
    for line in f:
        temp = line.strip('\n').split(' ')
        if len(temp) != expected_dim + 1:  # +1 because the first element is the UUID
                continue
        else:
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


if __name__ == "__main__":
    
    scenario='trace' # trace cadets  fivedirections  theia

    path = "/thayerfs/home/f006dg0/Normal/32/"
    filein = "{}{}_train_32.txt".format(path, scenario)
    fileout = "emb_{}_train_32.pickle".format(scenario)

    convert_embeddings(filein, fileout)

