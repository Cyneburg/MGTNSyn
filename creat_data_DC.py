import csv
import pandas as pd
from rdkit import Chem
import networkx as nx
from utils_test import *

import warnings
warnings.filterwarnings('ignore')


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

def edge_attr_features(bone):
    bond_featurizer = BondFeaturizer(
        allowable_sets={
            "bond_type": {"single", "double", "triple", "aromatic"},
            "conjugated": {True, False},
        }
    )
    return bond_featurizer.encode(bone)

def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),#获取原子名称
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #获取原子连接数（受H是否隐藏影响）
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +#获取氢原子总个数
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +#获取原子隐式化合价
                    [atom.GetIsAromatic()]+[atom.GetIsotope()])
            #矩阵维度：药物原子个数 * 药物特征维度


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile): #输出每个药物分子的：原子个数、每个原子的特征矩阵、分子图邻接表
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    app_atoms, atoms = [], []
    features = []
    lapos_edge,edges = [],[]
    edge_attr = []

    for bond in mol.GetBonds():#获取药物分子邻接表
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        lapos_edge.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_attr.append(edge_attr_features(bond))
        app_atoms.append(bond.GetBeginAtomIdx())
        app_atoms.append(bond.GetEndAtomIdx())

    for atom in mol.GetAtoms():
        atoms.append(atom.GetIdx())
        feature = atom_features(atom)#计算药物原子特征
        features.append(feature / sum(feature))
        if atom.GetIdx() not in app_atoms:#为独立原子提供特殊位置编码
            lapos_edge.append([atom.GetIdx(), mol.GetNumAtoms()])

    g = nx.Graph(edges).to_directed()
    g_lapos = nx.Graph(lapos_edge).to_directed()

    edge_index, lapos_edge_index = [], []

    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    for e1, e2 in g_lapos.edges:
        lapos_edge_index.append([e1, e2])

    return c_size, atoms, features, edge_index, lapos_edge_index, edge_attr


def creat_data(datasets, cellfile):
    file2 = cellfile
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    # print('cell_features', cell_features)

    compound_iso_smiles = []
    df = pd.read_csv('./data/smiles/smiles.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    # print('compound_iso_smiles', compound_iso_smiles)

    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
        # smile_graph[smile] = smile

    # convert to PyTorch data format
    processed_data_file_train = './data/processed/' + datasets + '_train.pt'

    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('./data/' + datasets + '.csv')
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
        # make data PyTorch Geometric ready
        print('开始创建数据')
        print(datasets)
        # print('cell_features', cell_features)
        TestbedDataset(root='./data/', dataset=datasets + '_drug1', xd=drug1, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_graph)
        TestbedDataset(root='./data/', dataset=datasets + '_drug2', xd=drug2, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_graph)
        print('创建数据成功')
        print('preparing ', datasets + '_.pt in pytorch format!')


if __name__ == "__main__":
    cellfile = './data/cell/independent_cell_features_954.csv'
    da = ['synergy_data_examples']
    for dataset in da:
        creat_data(dataset, cellfile)
