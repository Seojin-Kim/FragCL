import argparse
import json
import os
import pickle
import random
from itertools import repeat
from os.path import join

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

#from molecule_datasets import allowable_features
from rdkit.Chem.AllChem import ReactionFromSmarts

from rdkit.Chem import AllChem, Descriptors

allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_fragment(mol):
    if mol is None:
        print('something wrong')
    Rxn = ReactionFromSmarts('[C:1]-!@[C:2]>>[C:1].[C:2]')
    fragments = Rxn.RunReactants([mol])
    reactions = []
    for (f1, f2) in fragments:
        frag1 = Chem.MolToSmiles(f1)
        frag2 = Chem.MolToSmiles(f2)
        if set([frag1, frag2]) not in reactions:
            reactions.append(set([frag1, frag2]))
    
    min_frag_size_diff = -1
    balanced_rxn = None

    for rxn_set in reactions:
        rxn_list = list(rxn_set)
        if len(rxn_list) != 2:
            continue
        if abs(len(rxn_list[0]) - len(rxn_list[1])) < min_frag_size_diff or min_frag_size_diff < 0:
            balanced_rxn = rxn_list
            min_frag_size_diff = abs(len(rxn_list[0]) - len(rxn_list[1]))

    

    #if balanced_rxn is not None:
    #    if balanced_rxn[0].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
    #        return None
    #    elif balanced_rxn[1].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
    #        return None
    if balanced_rxn is None:
        return None
    mol1 = Chem.MolFromSmiles(balanced_rxn[0])
    mol2 = Chem.MolFromSmiles(balanced_rxn[1])

    return mol1, mol2
    
            

def mol_combination(mol1, mol2):
    Rxn = ReactionFromSmarts('[*;!H0:1].[*;!H0:2]>>[*:1]-[*:2]')
    combination = Rxn.RunReactants([mol1, mol2])
    if combination is None:
        raise 'combination error'
    else:
        return combination[0][0]
    

def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 2  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data






def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
        
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data




def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
        
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    #conformer = mol.GetConformers()[0]
    #positions = conformer.GetPositions()
    #positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=None)
    return data


def summarise():
    """ summarise the stats of molecules and conformers """
    dir_name = '{}/rdkit_folder'.format(data_folder)
    drugs_file = '{}/summary_drugs.json'.format(dir_name)

    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    # expected: 304,466 molecules
    print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

    sum_list = []
    drugs_summary = list(drugs_summary.items())

    for smiles, sub_dic in tqdm(drugs_summary):
        ##### Path should match #####
        if sub_dic.get('pickle_path', '') == '':
            continue


        mol_path = join(dir_name, sub_dic['pickle_path'])
        with open(mol_path, 'rb') as f:
            mol_sum = {}
            mol_dic = pickle.load(f)
            conformer_list = mol_dic['conformers']
            conformer_dict = conformer_list[0]
            rdkit_mol = conformer_dict['rd_mol']
            data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

            mol_sum['geom_id'] = conformer_dict['geom_id']
            mol_sum['num_edge'] = len(data.edge_attr)
            mol_sum['num_node'] = len(data.positions)
            mol_sum['num_conf'] = len(conformer_list)

            # conf['boltzmannweight'] a float for the conformer (a few rotamers)
            # conf['conformerweights'] a list of fine weights of each rotamer
            bw_ls = []
            for conf in conformer_list:
                bw_ls.append(conf['boltzmannweight'])
            mol_sum['boltzmann_weight'] = bw_ls
        sum_list.append(mol_sum)
    return sum_list


class MoleculeDatasetFrag(InMemoryDataset):

    def __init__(self, root, transform=None, seed=777,
                 pre_transform=None, pre_filter=None, empty=False, **kwargs):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)


        self.root, self.seed = root, seed
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(MoleculeDatasetFrag, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.data1, self.slices1 = torch.load(self.processed_paths[0] + "_1")
            self.data2, self.slices2 = torch.load(self.processed_paths[0] + "_2")

        
        #print('root: {},\ndata: {},\nn_mol: {},\n'.format(
        #    self.root, self.data, self.n_mol))

    def get(self, idx):
        data = Data()
        data1 = Data()
        data2 = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]

            item1, slices1 = self.data1[key], self.slices1[key]
            s = list(repeat(slice(None), item.dim()))
            s[data1.__cat_dim__(key, item1)] = slice(slices1[idx], slices1[idx+1])
            data1[key] = item1[s]

            item2, slices2 = self.data2[key], self.slices2[key]
            s = list(repeat(slice(None), item.dim()))
            s[data2.__cat_dim__(key, item2)] = slice(slices2[idx], slices2[idx+1])
            data2[key] = item2[s]
        return data, data1, data2

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        data1_list = []
        data2_list = []
        data_smiles_list = []
        one_functional_group_list = []

        
        data_list = []
        data_smiles_list = []
        input_path = "/home/osikjs/GraphMVP/datasets/molecule_datasets/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
        input_df = pd.read_csv(input_path, sep=',',
                                compression='gzip',
                                dtype='str')
        zinc_id_list = list(input_df['zinc_id'])
        smiles_list = list(input_df['smiles'])

        err_cnt = 0
        id = 0
        for i in range(len(smiles_list)):
            print(i)
            s = smiles_list[i]
            # each example contains a single species
            try:
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol is not None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    
                    mols = mol_fragment(rdkit_mol)

                    if mols is None:
                        err_cnt += 1
                        continue
                        #rdkit_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rdkit_mol))
                        #Atoms = rdkit_mol.GetAtoms()
                        #minh = 0
                        #for atom in Atoms:
                        #    attached_h = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
                        #    if atom.GetAtomicNum() == 6 and minh < attached_h:
                        #        minh = attached_h
                        #if minh == 0:
                        #    continue
                        #else:
                        #    one_functional_group_list.append(rdkit_mol)
                        #    continue                     
                    else:
                        id += 1
                        data1 = mol_to_graph_data_obj_simple(mols[0])
                        data2 = mol_to_graph_data_obj_simple(mols[1])

                    data = mol_to_graph_data_obj_simple(rdkit_mol)

                    data.mol_id = torch.tensor([id])
                    data.id = torch.tensor([id])
                    data_list.append(data)
                    data1.mol_id = torch.tensor([id])
                    data1.id = torch.tensor([id])
                    data1_list.append(data1)
                    data2.mol_id = torch.tensor([id])
                    data2.id = torch.tensor([id])
                    data2_list.append(data2)
                    
                    

                    
                    # manually add mol id
                    #id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                    #data.id = torch.tensor([id])
                    # id here is zinc id value,
                    # stripped of leading zeros
                    #data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
            except:
                continue
        #for i in range(len(one_functional_group_list)//2):
        #    data1 = mol_to_graph_data_obj_simple(one_functional_group_list[2*i])
        #    data2 = mol_to_graph_data_obj_simple(one_functional_group_list[2*i+1])
        #    data = mol_to_graph_data_obj_simple(mol_combination(one_functional_group_list[2*i], one_functional_group_list[2*i+1]))

        #    data.mol_id = torch.tensor([id])
        #    data.id = torch.tensor([id])
        #    data_list.append(data)
        #    data1.mol_id = torch.tensor([id])
        #    data1.id = torch.tensor([id])
        #    data1_list.append(data1)
        #    data2.mol_id = torch.tensor([id])
        #    data2.id = torch.tensor([id])
        #    data2_list.append(data2)

        #    id += 1
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        data1, slices1 = self.collate(data1_list)
        data2, slices2 = self.collate(data2_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save((data1, slices1), self.processed_paths[0] + "_1")
        torch.save((data2, slices2), self.processed_paths[0] + "_2")

        #print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % id)
        return


def load_SMILES_list(file_path):
    SMILES_list = []
    with open(file_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            SMILES_list.append(line.strip().decode())
    return SMILES_list


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', type=bool, default=False, help='cal dataset stats')
    parser.add_argument('--n_mol', type=int, help='number of unique smiles/molecules')
    parser.add_argument('--data_folder', type=str)
    args = parser.parse_args()

    data_folder = args.data_folder

    if args.sum:
        sum_list = summarise()
        with open('{}/summarise.json'.format(data_folder), 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol = args.n_mol
        root = '{}/molecule_datasets/zinc_standard_agent_keep_branch'.format(data_folder)

        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        #Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)
        # Generate 2D Datasets (2D SMILES)
        MoleculeDatasetFrag(root=root)
    
    ##### to data copy to SLURM_TMPDIR under the `datasets` folder #####
    '''
    wget https://dataverse.harvard.edu/api/access/datafile/4327252
    mv 4327252 rdkit_folder.tar.gz
    cp rdkit_folder.tar.gz $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    tar -xvf rdkit_folder.tar.gz
    '''

    ##### for data pre-processing #####
    '''
    python GEOM_dataset_preparation.py --n_mol 100 --n_conf 5 --n_upper 1000 --data_folder $SLURM_TMPDIR
    python GEOM_dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000 --data_folder $SLURM_TMPDIR
    '''
