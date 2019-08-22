import argparse
import torch
import pickle
import os
from scipy import sparse
import numpy as np
from utils import naive_sparse2tensor
import h5py
from data import create_sparse_matrix, create_bow

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--input_file', type=str, default='data/processed_output.h5',
                    help='processed output h5 location')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Load the best saved model.
with open(args.save, 'rb') as f:
    device = torch.device("cpu")

    model = torch.load(f).to(device)
    model.eval()
    input_dim = model.in_dim

    #Load mappings
    processed_f = h5py.File(args.input_file, 'r')
    author2idx = processed_f['author2idx']
    idx2author = processed_f['idx2author']
    authors_n = len(author2idx)

    authors = None
    while authors != "bye":
        # text_author_id_list = input("Give ids of authors, separated by space:")
        # Hard coded - debug
        text_author_id_list = ' '.join([str(author_id) for author_id in list(author2idx.keys())[:1]])

        authors = [author_id for author_id in text_author_id_list.strip().split()]

        author_idxs = [author2idx[author][0] for author in authors]

        paper2author_idxs_scores = {}
        paper2author_idxs_scores['777'] = {
            'author_idxs':author_idxs,
            'scores': [1]
        }

        array, offsets, weights = create_bow(paper2author_idxs_scores)

        array = array.to(device)
        offsets = offsets.to(device)
        weights = weights.to(device)

        recon_batch, mu, logvar = model(array, offsets, weights)

        # TODO: May use softmax on 'recon_batch' to get probabilities!
        TOP = 5
        recommended_author_ids = [idx2author[str(author_id)][0].decode("utf-8") for author_id in recon_batch.detach().numpy().flatten().argsort()[-TOP:][::-1]]
        print("Top {} authors for {}: {}".format(TOP, text_author_id_list, recommended_author_ids))
        # Hard coded - debug
        break
    # TODO: Test query by title (later stage)
    # title = None
    # while title != "bye":
    #     title = input("Enter a paper's title")

