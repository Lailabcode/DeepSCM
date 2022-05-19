# DeepSCM
A convolutional neural network model to predict spatial charge map (SCM) score, a molecular dynamics simulation-based model to predict antibody viscosity. 

# Prepare working environment
The working environment can be setup using the following commands:

```
conda create -n deepscm python=3.7
source activate deepscm
conda install scikit-learn
pip install --upgrade tensorflow

cd deepscm-master
```

# Prediction
There are two steps before running the DeepSCM prediction. 
First, prepare two fasta files of the heavy chain and light chain, respectively. 
The fasta files must have the following format:

seq_H.fasta
```
>name_1	
[heavy chain sequence]
>name_2
[heavy chain sequence]
>name_3
[heavy chain sequence]
```
seq_L.fasta
```
>name_1	
[light chain sequence]
>name_2
[light chain sequence]
>name_3
[light chain sequence]
```
Using the ANARCI program (https://github.com/oxpig/ANARCI) to number the antibody sequences using IMGT definition in a CSV format. 
```
ANARCI -i seq_H.fasta -o seq_aligned -s imgt -r heavy --csv
ANARCI -i seq_L.fasta -o seq_aligned -s imgt -r light --csv
```
Second, run the sequence preprocessing program to generate the input file for DeepSCM.
```
python seq_preprocessing.py --seq_H seq_aligned_H.csv --seq_L seq_aligned_KL.csv --outfile DeepSCM_input.txt
```
Finally, the prediction can be run
```
python pred.py --infile DeepSCM_input.txt
```

# Citation
PK Lai, DeepSCM: An efficient convolutional neural network surrogate model for the screening of therapeutic antibody viscosity, 
Comput. Struct. Biotechnol. J., 20:2143-2152, 2022 (https://doi.org/10.1016/j.csbj.2022.04.035).

# Commerical use
If you wish to obtain Software for any commercial purposes, you will need to execute a separate licensing agreement and pay a fee. In that case please contact: plai3@stevens.edu
