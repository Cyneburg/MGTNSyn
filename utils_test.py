import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
import torch_geometric.utils as utils
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.data import Data
import torch
import hashlib
from tqdm import trange

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in trange(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            # convert SMILES to molecular representation using rdkit
            c_size, atoms, features, edge_index, lapos_edge_index, edge_attr = smile_graph[smiles]
            
            # wl_positional_encoding
            wlpos_encoding = wl_positional_encoding(atoms, edge_index)
            # laplacian_positional_encoding
            lapos_edge_index = torch.LongTensor(lapos_edge_index).transpose(1, 0)
            laplacian_, lap_weight = utils.get_laplacian(lapos_edge_index, normalization="sym")
            laplacian = utils.to_scipy_sparse_matrix(laplacian_)
            EigVal, EigVec = np.linalg.eig(laplacian.toarray())
            laplacian_positional_encoding = torch.from_numpy(EigVec).float()

            # get cell features
            cell = self.get_cell_feature(target, xt_featrue)
            if cell == False : # 如果读取cell失败则中断程序
                print('cell', cell)
                sys.exit()

            new_cell = []
            for n in cell:
                new_cell.append(float(n))

            GCNData = Data(x=torch.Tensor(features),
                           edge_index=torch.LongTensor(edge_index).transpose(-1, 0),
                           edge_attr=torch.tensor(edge_attr, dtype=torch.int64),
                           lapos_encoding=laplacian_positional_encoding,
                           wlpos_encoding = wlpos_encoding,
                           y=torch.FloatTensor([labels]),
                           cell=torch.FloatTensor([new_cell]),
                           c_size=torch.LongTensor([c_size])
                           )
           
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        torch.save(self.collate(data_list), self.processed_paths[0])

def wl_positional_encoding(node_list,edge_list):
    """
           WL-based absolute positional embedding
           adapted from

           "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
           Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
           https://github.com/jwzhanggy/Graph-Bert
       """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair[0], pair[1]
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    return torch.LongTensor(list(node_color_dict.values()))

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = stats.pearsonr(y, f)[0]
    return rp
def p_value(y,f):
    rp = stats.pearsonr(y, f)[1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci