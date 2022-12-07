# Instructions

Authors: Shengchao Liu, Hanchen Wang, Weiyang Liu, Joan Lasenby, Hongyu Guo, Jian Tang

## Environments
Install packages under conda env
```bash
conda create -n GraphMVP python=3.7
conda activate GraphMVP

conda install -y -c rdkit rdkit
conda install -y -c pytorch pytorch=1.9.1
conda install -y numpy networkx scikit-learn
pip install ase
pip install git+https://github.com/bp-kelley/descriptastorus
pip install ogb
export TORCH=1.9.0
export CUDA=cu102  # cu102, cu110

wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.7.2
```

Or just use

```
conda env create -f fragcl.yaml
```


## Dataset Preprocessing
First of all, place your SMILES file at **datasets/smiles.csv**
For data preprocessing, please use the following commands:
```
cd src_classification
python GEOM_dataset_preparation_frag_allsingle.py --data_folder ../datasets
cd ..
```

## Experiments
For pretraining, please use the followng commands:
```
cd scripts_classification
bash submit_pretraining_fragcl.sh
```

Then, the model will be saved at **output** folder


