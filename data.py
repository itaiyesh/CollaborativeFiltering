import gc
import json
import os
import warnings

import h5py
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import argparse

from utils import DatasetBuffer

# TODO: Put this class elsewhere
MAXIMUM_LEN = 30  # TODO: Get this parameter from preprocessing


class H5Dataset(Dataset):
    def __init__(self, interaction_file_path, embeddings_file_path, group_name, ds_name):
        self.interaction_file_path = interaction_file_path
        self.embeddings_file_path = embeddings_file_path
        self.group_name = group_name
        self.ds_name = ds_name
        self.authors_n = None
        self.maximum_len = MAXIMUM_LEN

        with h5py.File(self.interaction_file_path, 'r', libver='latest') as f:  # , swmr=True)
            samples = f[group_name][ds_name]
            self.n = len(samples)
            self.authors_n = len(f['author2idx'])

    def __len__(self):
        return self.n

    def __getitem__(self, indx):
        return indx  # TODO: Override dataloader to avoid having to call this


def create_sparse_matrix( authors_n, paper2author_idxs_scores, sorted_keys):
    # TODO: You may want to cache this (based on index range)
    rows = []
    cols = []
    scores = []
    for i, paper_id in enumerate(sorted_keys):
        paper_values = paper2author_idxs_scores[paper_id]
        for author_idx, score in zip(paper_values['author_idxs'], paper_values['scores']):
            rows.append(i)
            cols.append(author_idx)
            scores.append(score)

    data = sparse.csr_matrix((scores,
                              (rows, cols)), dtype='float64',
                             shape=(len(sorted_keys), authors_n))

    return data


def create_bow(paper2author_idxs_scores, sorted_keys = None):
    # TODO: We may use paper2idxs calculated in preprocessing (+ add batch offset)
    if sorted_keys is None:
        warnings.warn("Using random order for paper indexing in batch")
        sorted_keys = list(paper2author_idxs_scores.keys())

    array = []
    offsets = []
    weights = []
    last_offset = 0

    for paper_id in sorted_keys:
        paper_values = paper2author_idxs_scores[paper_id]
        author_idxs = paper_values['author_idxs']
        scores = paper_values['scores']

        offsets.append(last_offset)
        array += author_idxs
        weights += scores
        last_offset += len(author_idxs)

    # TODO: Calculate weight tensor from data
    array = torch.tensor(array, dtype=torch.long)
    offsets = torch.tensor(offsets, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float)

    return array, offsets, weights


class H5DataLoader(DataLoader):
    def __init__(self, dataset, sampler, batch_size, num_workers):
        super(H5DataLoader, self).__init__(dataset=dataset,
                                           sampler=sampler,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           # TODO: Do we want this drop last?
                                           drop_last=False,
                                           collate_fn=self.collate_fn
                                           )
        self.dataset = dataset

    def collate_fn(self, index_range):
        with h5py.File(self.dataset.interaction_file_path, 'r', libver='latest') as f, h5py.File(self.dataset.embeddings_file_path, 'r', libver='latest') as embs_f:  # , swmr=True)
            ds = f[self.dataset.group_name][self.dataset.ds_name][index_range[0]: index_range[-1] + 1]
            paper_ids = ds[:, 0]
            author_idxs = ds[:, 1]  # Get all author idxs in batch as a list
            scores = ds[:, 2]

            # TODO: We calculate sparse matrixes twice!
            paper2author_idxs_scores = {}
            for paper_id, author_idx, score in zip(paper_ids, author_idxs, scores):
                if paper_id not in paper2author_idxs_scores:
                    paper2author_idxs_scores[paper_id] = {}
                    paper2author_idxs_scores[paper_id]['author_idxs'] = []
                    paper2author_idxs_scores[paper_id]['scores'] = []

                paper2author_idxs_scores[paper_id]['author_idxs'].append(int(author_idx))
                paper2author_idxs_scores[paper_id]['scores'].append(float(score))

            # Sort paper2author_idxs_scores by paper2embedding_idx before everything.
            # TODO: Cache this (can't put in dataloder...may instantiate many times?)
            paper2embedding_idx = embs_f['paper2embeddings_idx']
            paper_embedding_idxs = [paper2embedding_idx[paper_id][()][0] for paper_id in paper2author_idxs_scores.keys()]
            paper_embedding_idxs, sorted_keys = zip(*sorted(zip(paper_embedding_idxs, paper2author_idxs_scores.keys()), key = lambda x: x[0]))

            sorted_keys = list(sorted_keys)
            paper_embedding_idxs = list(paper_embedding_idxs)

            # Important: The order of keys in paper2idxs is crucial for coordination between the two:
            array, offsets, weights = create_bow(paper2author_idxs_scores, sorted_keys)
            embs = embs_f['embeddings'][paper_embedding_idxs]
            embs = torch.tensor(embs, dtype=torch.float)
            #TODO: Remove in inference (we don't use this matrix)
            sparse_array = create_sparse_matrix(self.dataset.authors_n, paper2author_idxs_scores,sorted_keys)

            return array, offsets, weights, embs, sparse_array

    # # Close ds file
    # def __del__(self):
    #     print self.id, 'died'


class RangeSampler(Sampler):
    def __init__(self, range):
        self.range = range

    def __iter__(self):
        return iter(self.range)

    def __len__(self):
        return len(self.range)

MINIMUM_YEAR = 2000
# LIMIT = 100000  # None#1000000
CHUNK_SIZE = 2048


def skip_paper(paper_json):
    return 'year' not in paper_json or paper_json['year'] < MINIMUM_YEAR

def create_paper_author_score_triples(json_file, output_file, LIMIT):
    triples = []
    AUTHOR_SCORE = 1.0
    CITED_AUTHOR_SCORE = 0.2
    paper2authors = {}
    warnings.warn("You're using a chunk of {}".format(CHUNK_SIZE))

    # cited_papers_pairs = []

    filtered_lines = 0
    total = 0

    print("Opening {}".format(json_file))
    with open(json_file, 'r') as input_f, h5py.File(output_file, 'w') as output_h5:

        string_dt = h5py.special_dtype(vlen=str)

        # Paper -> [author1,...authorN]
        # paper2authors_ds = output_h5.create_dataset('paper2author',
        #                                             maxshape=(500000000, 2),
        #                                             shape=(0, 2),
        #                                             chunks=(CHUNK_SIZE, 2),
        #                                             dtype=string_dt)

        paper_direct_author_score_ds = output_h5.create_dataset('paper_direct_author_score',
                                                                # TODO: Change big size to None
                                                                maxshape=(150000000, 3),
                                                                # compression="gzip",
                                                                shape=(0, 3),

                                                                chunks=(CHUNK_SIZE, 3),
                                                                dtype=string_dt)

        cited_paper_pairs_ds = output_h5.create_dataset('cited_paper_pairs',
                                                        maxshape=(150000000, 2),
                                                        shape=(0, 2),
                                                        chunks=(CHUNK_SIZE, 2),
                                                        dtype=string_dt)

        # This is the final ds (summed over all scores)
        paper_author_score_ds = output_h5.create_dataset('paper_author_score',
                                                         maxshape=(150000000, 3),
                                                         shape=(0, 3),
                                                         chunks=(CHUNK_SIZE, 3),
                                                         dtype=string_dt)

        # tuple2score_grp = output_h5.create_group('tuple2score')
        paper2authors_grp = output_h5.create_group('paper2authors')

        # paper2authors_bf = DatasetBuffer(paper2authors_ds, buffer_size=CHUNK_SIZE)
        paper_direct_author_bf = DatasetBuffer(paper_direct_author_score_ds, buffer_size=CHUNK_SIZE)
        cited_paper_pairs_bf = DatasetBuffer(cited_paper_pairs_ds, buffer_size=CHUNK_SIZE)
        paper_author_score_bf = DatasetBuffer(paper_author_score_ds, buffer_size=CHUNK_SIZE)

        total_lines = LIMIT if LIMIT else 4107340
        pbar = tqdm(total=total_lines, desc='Collecting data')
        continue_read = True
        while continue_read:

            lines = input_f.readlines(8192)

            if not lines:
                break
            for line in lines:

                total += 1

                if total > total_lines:
                    continue_read = False
                    break

                pbar.update(1)

                paper_json = json.loads(line)

                if skip_paper(paper_json):
                    filtered_lines += 1
                    continue

                paper_id = paper_json['id']

                # paper2authors[paper_id] = []

                paper_authors = []

                for author in paper_json['authors']:
                    author_id = author['id']

                    # # # TODO: Sanity check:
                    # if author_id == '1042385918':
                    #     print(paper_json['authors'])

                    # paper2authors[paper_id].append(author_id)
                    paper_authors.append(author_id)

                    # triples.append((paper_id, author_id, AUTHOR_SCORE))
                    paper_direct_author_bf.add(np.array([paper_id, author_id, AUTHOR_SCORE], dtype=object))

                paper2authors_grp.create_dataset(paper_id, data=np.array(paper_authors, dtype='S'), dtype=string_dt)

                if 'references' in paper_json:
                    for ref_id in paper_json['references']:
                        # cited_papers_pairs.append((paper_id, ref_id))
                        cited_paper_pairs_bf.add(np.array([paper_id, ref_id], dtype=object))
        paper_direct_author_bf.close()
        cited_paper_pairs_bf.close()
        pbar.close()

        print("Total papers in db: {}. filtered {}: {}".format(total, MINIMUM_YEAR, filtered_lines))

        # Sum all scores per (paper,author, score).
        # (paper, author) -> total score
        # This is the only object held in memory throughout processing
        tuple2score = {}
        for paper, author, score in tqdm(paper_direct_author_score_ds, total=len(paper_direct_author_score_ds),
                                         desc='Summing direct authors'):
            if (paper, author) not in tuple2score:
                tuple2score[(paper, author)] = 0

            tuple2score[(paper, author)] += float(score)

        # Add all cited author triples ( can only be done in 2nd iteration)
        # This stage is a bottleneck in the processing since we access paper2author mapping (which is on disk) frequently.
        # better to move paper2author map to memory on stronger systems (I promise a 4x speed up).
        for paper, cited_paper in tqdm(cited_paper_pairs_ds, total=len(cited_paper_pairs_ds),
                                       desc='Summing cited authors'):

            # if cited_paper not in paper2authors:
            if cited_paper not in paper2authors_grp:
                # This case is only possible when not training on full dataset,
                # otherwise, cited_paper is expected to be in paper2author hash.
                continue
            # for cited_author in paper2authors[cited_paper]:
            for cited_author in paper2authors_grp[cited_paper]:
                if (paper, cited_author) not in tuple2score:
                    tuple2score[(paper, cited_author)] = 0

                tuple2score[(paper, cited_author)] += CITED_AUTHOR_SCORE

        for (paper_id, author_id), score in tqdm(tuple2score.items(), desc='Writing (paper, author)->score to h5'):
            paper_author_score_bf.add(np.array([paper_id, author_id, score], dtype=object))

        paper_author_score_bf.close()

        # Evict space
        del tuple2score
        del paper_direct_author_score_ds
        del cited_paper_pairs_ds
        del paper2authors_grp

        gc.collect()


def unique(data_file, column, indexes=None):
    vals = set()
    with h5py.File(data_file, 'r') as f:
        if indexes:
            print("Sorting {} instances".format(len(indexes)))
            ds = f['paper_author_score'][sorted(indexes)]
        else:
            ds = f['paper_author_score']
        for row in tqdm(ds, total=len(ds), desc='Computing unique instances for column {}'.format(column)):
            vals.add(row[column])
    return vals


def within_range(val, range):
    return val >= range[0] and val < range[1]


def add_data(paper_author_score_ds, tr_bf, te_bf, index_range, paper2idxs, author2idx, test_proportion=None):
    def row_convert_author2idx(row):
        row_copy = row
        row_copy[1] = author2idx[row[1]]
        return row_copy

    # If no test, just add everything to training (as in the case for Train dataset)
    if test_proportion is None:
        all_indexes = [item for sublist in list(paper2idxs.values())[index_range[0]:index_range[1]] for item in sublist]
        _list = paper_author_score_ds[sorted(all_indexes)]
        for row in tqdm(_list, total=len(_list), desc='Adding all instances to train.'):
            tr_bf.add(row_convert_author2idx(row))

        return

    # Group by paper and split
    _list = list(paper2idxs.items())[index_range[0]: index_range[1]]
    for paper, indexes in tqdm(_list, total=len(_list), desc='Iterating instances and splitting train/test'):
        indexes = sorted(indexes)
        indexes_n = len(indexes)

        if len(indexes) >= 5:
            sample_test_indexes = np.random.choice(indexes, size=int(test_proportion * indexes_n), replace=False)
            sample_train_indexes = list(set(indexes) - set(sample_test_indexes))

            sample_test_indexes = sorted(list(sample_test_indexes))
            sample_train_indexes = sorted(sample_train_indexes)

            for row in paper_author_score_ds[sample_train_indexes]:
                if row[1] in author2idx:
                    tr_bf.add(row_convert_author2idx(row))

            for row in paper_author_score_ds[sample_test_indexes]:
                if row[1] in author2idx:
                    te_bf.add(row_convert_author2idx(row))
        else:
            for row in paper_author_score_ds[indexes]:
                if row[1] in author2idx:
                    tr_bf.add(row_convert_author2idx(row))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
    parser.add_argument('--json_file', type=str, default='C:\\Users\iyeshuru\Downloads\dblp_papers_v11.txt',
                        help='Processed input h5 file.')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Limit number of data to process.')
    args = parser.parse_args()

    # json_file = 'C:\\Users\iyeshuru\PycharmProjects\PapersProject\\flow\dblp.cut'
    # json_file = 'C:\\Users\iyeshuru\PycharmProjects\PapersProject\\flow\dblp.large.cut'
    # json_file = 'C:\\Users\iyeshuru\Downloads\dblp_papers_v11.txt'
    # json_file = 'C:\\Users\iyeshuru\PycharmProjects\PapersProject\\flow\dblp_test.txt'
    json_file = args.json_file

    DATA_DIR = 'data/'

    # index2paper
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    raw_output = os.path.join(DATA_DIR, 'raw_output.h5')
    processed_output_file = os.path.join(DATA_DIR, 'processed_output.h5')
    embeddings_output_file = os.path.join(DATA_DIR, 'embeddings_output.h5')


    # Process data
    create_paper_author_score_triples(json_file, raw_output, args.limit)

    paper2idxs = {}

    # Create paper id to indexes (group)
    warnings.warn("Loading all paper IDs into memory.")
    with h5py.File(raw_output, 'r') as f:
        for i, (paper, _, _) in tqdm(enumerate(f['paper_author_score']), total=len(f['paper_author_score']),
                                     desc='Building paper2idxs mapping'):
            if paper not in paper2idxs:
                paper2idxs[paper] = []
            paper2idxs[paper].append(i)

    unique_papers_count = len(paper2idxs.keys())
    n_heldout_users = int(unique_papers_count * 0.2)

    # Split Train/Validation/Test User Indices
    tr_papers_index_range = [0, unique_papers_count - n_heldout_users * 2]
    vd_papers_index_range = [unique_papers_count - n_heldout_users * 2, unique_papers_count - n_heldout_users]
    te_papers_index_range = [unique_papers_count - n_heldout_users, unique_papers_count]

    for dataset, index_range in zip([
        "Train", "Validation", "Test"
    ], [
        tr_papers_index_range,
        vd_papers_index_range,
        te_papers_index_range,

    ]):
        print("{} papers: {}".format(dataset, index_range[1] - index_range[0]))

    warnings.warn("Loading all paper indexes into memory.")
    ranges = [tr_papers_index_range, vd_papers_index_range, te_papers_index_range]

    tr_indexes = [item for sublist in list(paper2idxs.values())[tr_papers_index_range[0]:tr_papers_index_range[1]] for
                  item in sublist]

    unique_train_authors = unique(raw_output, 1, tr_indexes)
    unique_train_authors_count = len(unique_train_authors)

    author2idx = dict((pid, i) for (i, pid) in enumerate(unique_train_authors))

    with h5py.File(raw_output, 'r') as raw_f, h5py.File(processed_output_file, 'w') as processed_f:

        ################## Add Paper, author, scores data ##############################
        paper_author_score_ds = raw_f['paper_author_score']

        # Save mapping to ds
        author2idx_grp = processed_f.create_group('author2idx')
        for author, idx in tqdm(author2idx.items(), total=len(author2idx), desc='Saving mapping'):
            author2idx_grp.create_dataset(author,
                                          data=(idx,))  # data=np.array(paper_authors, dtype='S'), dtype=string_dt)

        # Save reverse mapping to ds (used in inference)
        idx2author_grp = processed_f.create_group('idx2author')
        for author, idx in tqdm(author2idx.items(), total=len(author2idx), desc='Saving reverse mapping'):
            idx2author_grp.create_dataset(str(idx),
                                          data=np.array([author], dtype='S'))

        train_grp = processed_f.create_group('train')
        validation_grp = processed_f.create_group('validation')
        test_grp = processed_f.create_group('test')

        # Creating datasets and buffers for all train/validation/test ->train/test combinations
        list_pairs = list(map(
            lambda grp:
            [
                DatasetBuffer(grp.create_dataset('train',
                                                 maxshape=paper_author_score_ds.shape,
                                                 shape=(0, paper_author_score_ds.shape[1]),
                                                 chunks=paper_author_score_ds.chunks,
                                                 dtype=paper_author_score_ds.dtype
                                                 )),
                DatasetBuffer(grp.create_dataset('test',
                                                 maxshape=paper_author_score_ds.shape,
                                                 shape=(0, paper_author_score_ds.shape[1]),
                                                 chunks=paper_author_score_ds.chunks,
                                                 dtype=paper_author_score_ds.dtype
                                                 ))
            ]
            , [train_grp, validation_grp, test_grp]))

        buffers = [item for sublist in list_pairs for item in sublist]
        # tr_tr, tr_te, val_tr, val_te, test_te, test_tr = buffers

        # Filter by author, group by paper and split by proportion.
        for i, ((tr_tr, tr_te), index_range) in enumerate(zip(list_pairs, ranges)):
            print("Building dataset: {}".format(['Train', 'Validation', 'Test'][i]))
            # indexes = [item for sublist in list(paper2idxs.items())[index_range[0]:index_range[1]] for item in sublist]
            add_data(paper_author_score_ds, tr_tr, tr_te, index_range, paper2idxs, author2idx,
                     test_proportion=None if i == 0 else 0.2)

        for buffer in buffers:
            buffer.close()


    ########## Add title embeddings #######################
    #TODO: Pass last two args differnetly...
    from get_embeddings import collect_embeddings
    # collect_embeddings(json_file, embeddings_output_file, args.limit, CHUNK_SIZE, skip_paper)
    collect_embeddings(json_file, unique_papers_count, embeddings_output_file, args.limit, CHUNK_SIZE, skip_paper)



    print("Done!")
