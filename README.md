# Scoring-Assisted Generative Exploration for Proteins (SAGE-Prot)
=======================
![Figure 1](https://github.com/user-attachments/assets/2d0297ac-1382-437f-941c-868f158b0a35)

What is SAGE-Prot?
----------------
**Scoring-Assisted Generative Exploration for Proteins (SAGE-Prot)** is an effective framework for generating high-scoring proteins with deep neural networks, genetic algorithms, retrieval-augmented generation, and various QSAR/QSPR scoring models for desired objectives.

All source codes will be uploaded.

# Notes
This is additional supplementary data in "Scoring-Assisted Generative Exploration for Proteins (SAGE-Prot): A Framework for Multi-Objective Protein Optimization via Iterative Sequence Generation and Evaluation"

Prerequisites
-------------
* LINUX/UNIX Cluster Machines (Ubuntu 20.04)
* Python 3.9
* conda package: openbabel, hmmer
* pip package: numpy, scipy, scikit-learn, xgboost, nltk, networkx, pyyaml, nbformat, lightgbm, optuna, pandas, PyTDC, mordred, gensim, PubChemPy, torch, matplotlib, seaborn, jupyter, neptune-client, tqdm, rdkit-pypi tensorflow, deepchem, meeko, autograd, tensorflow_addons, tensorflow_probability, holoviews, Flask, Jinja2, bokeh, panel, guacamol, requests, transformers, sentencepiece, catboost, omegaconf, mlxtend, huggingface-hub, portalocker, ftfy, datasets, biotite, biopython, py3Dmol, torchensemble, ema-pytorch, einops, accelerate, x-transformers, aiohttp, attrs, jsonschema, charset_normalizer, swagger-spec-validator, torch-geometric, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, torchvision, [ESM](https://github.com/facebookresearch/esm), protobuf, tape_proteins, torchtext

How to Get Swiss-Prot Dataset
-------------
Please download Reviewed (Swiss-Prot) sequences at UniProt (https://www.uniprot.org/help/downloads)

How to Pretrain NLP Models (Example: LSTM)
-------------
python run_pretrain.py --model_type LSTM --dataset swissprot_reduced --dataset_path ./ --max_sequences_length 512 --hidden_size 256 --dropout 0.2 --n_layers 3 --num_epochs 150 --batch_size 1024 --save_root ./swissprot_reduced/ --seed 404 > SAGE_Protein_Pretrain_LSTM_swissprot_reduced.log

How to Reproduce Benchmark Results (Example: 00)
-------------
python run_optimization.py --model_type LSTM --apprentice_sampling_batch_size 8192 --expert_sampling_batch_size 8192 --benchmark_id 0 --dataset benchmark --dataset_path ./swissprot_reduced.txt --apprentice_load_dir ./swissprot_reduced/LSTM  --num_steps 100 --max_sequences_length 512 --apprentice_training_batch_size 256 --genetic_expert SEQUENCES > SAGE_PROTEIN_OPT_LSTM_GA_ID_00_swissprot_reduced.log

Contact Person
--------------
* Dr. Hocheol Lim (ihc0213@yonsei.ac.kr)

Acknowledgments
---------------
The research was supported by the Ministry of Trade, Industry, and Energy (MOTIE), 
the Republic of Korea, under the project “Industrial Technology Infrastructure 
Program” (Project No. RS-2024-00466693).

How to Cite
----------
Lim, Hocheol. "Scoring-Assisted Generative Exploration for Proteins (SAGE-Prot): A Framework for Multi-Objective Protein Optimization via Iterative Sequence Generation and Evaluation"
