import torch
import numpy as np

def sparse2torch_sparse(data, half_precision=False):

    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    # # np.exp(x) / sum(np.exp(x))
    # print(data)
    # print(type(data))
    # print(data.expm1())
    # exit()
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    # print(t.to_dense())
    # sd = data.toarray()
    # print(sd / np.linalg.norm(sd, ord=2, axis=-1)[:, np.newaxis])
    # exit()
    return t

def sparse2torch_sparseORIG(data, half_precision=False):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    if half_precision:
        # Added half precision here
        t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float().half(), [samples, features])
    else:
        t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])

    return t

def naive_sparse2tensor(data, half_precision=False):
    if half_precision:
        # Added half precision here
        t = torch.FloatTensor(data.toarray()).half()
    else:
        t = torch.FloatTensor(data.toarray())

    return t


# Dataset writing buffer to speed up writing to HDF5 file.
class DatasetBuffer:
    def __init__(self, dataset, buffer_size=2048):
        self.index = 0
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)
        self.count += 1
        if self.count == self.buffer_size:
            self.dataset.resize(self.dataset.shape[0] + self.buffer_size, axis=0)
            self.dataset[self.index:self.index + self.buffer_size] = np.vstack(self.buffer)

            self.count = 0

            self.index += self.buffer_size
            self.buffer = []

    # Adding remaining elements upon file closing
    def close(self):
        if self.count > 0:
            self.dataset.resize(self.dataset.shape[0] + self.count, axis=0)
            self.dataset[self.index:self.index + self.count] = np.vstack(self.buffer)
            self.index += self.count
            self.count = 0
            self.buffer = []

import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])