# DeepSCM
A convolutional neural network model to predict spatial charge map (SCM) score, a molecular dynamics simulation-based model to predict antibody viscosity. 

# Prepare working environment
The working environment can be loaded via the provided environment.yml file using the following commands:

```
conda env create -f environment.yml
source activate deepscm
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

