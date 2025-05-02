"""
Copyright (c) 2025 Hocheol Lim.
"""

import random
from typing import Optional

import numpy as np
from sage_prot.scoring.filters import clean_sequence

def find_homolog_seq(seq, num_trials=10, max_seq_len=512, prot_db='landmark'):
    import os
    import uuid
    import subprocess
    import pandas as pd
    
    init = 0
    flag_error = True
    
    BLAST_DB = prot_db
    BLAST_PATH = os.environ.get('BLAST_HOME')
    if BLAST_PATH is None:
        BLAST_PATH = '/opt/blast'
    
    BLAST_EXE = BLAST_PATH + '/ncbi-blast-2.15.0+/bin/'
    os.environ['BLASTDB'] = BLAST_PATH + '/' + BLAST_DB
    clean_seq = clean_sequence(seq)
    parent_seq = [clean_seq]
    output_seq = None
    
    while init < num_trials and flag_error:
        init = init + 1
        try:
            filename = str(uuid.uuid4()).replace("-","")
            filename_fasta = filename + ".fasta"
            filename_tsv = filename + ".tsv"
            
            f1 = open(filename_fasta, 'w')
            f1.write(">SAGE_"+filename+"\n" + clean_seq)
            f1.close()
            
            cmd_1 = BLAST_EXE + "blastp -query " + filename_fasta + " -db " + BLAST_DB + " -out " + filename_tsv + ' -outfmt "6 sseqid sseq length pident qcovs evalue" -num_threads 1 -max_target_seqs 10'
            cmd_1_flag = subprocess.run([cmd_1], shell=True)
            
            BLAST_OUT = pd.read_csv(filename_tsv, sep='\t', header=None)
            BLAST_OUT['Rank'] = BLAST_OUT.iloc[:, 3] * BLAST_OUT.iloc[:, 4] / 10000
            hit_homolgy_id = BLAST_OUT[BLAST_OUT['Rank'] != 1].sort_values(by='Rank', ascending=False).iloc[init-1].iloc[0]
            
            cmd_2 = BLAST_EXE + "blastdbcmd -db " + BLAST_DB + " -entry '" + hit_homolgy_id + "'"
            result = subprocess.run(cmd_2, shell=True, text=True, capture_output=True)
            lines = result.stdout.split('\n')
            output_seq = ''.join(lines[1:])
            
            output_seq = clean_sequence(output_seq)
            os.system('rm -rf '+filename+'*')
            
            if len(output_seq) <= max_seq_len and output_seq not in parent_seq:
                flag_error = False
            
        except:
            os.system('rm -rf '+filename+'*')
            flag_error = True
        return output_seq

def find_groups(aa, groups):
    possible_groups = []
    for group, amino_acids in groups.items():
        if aa in amino_acids:
            possible_groups.append(group)
    return possible_groups

def mutate_seq(seq, groups):
    clean_seq = clean_sequence(seq)
    mutation_position = random.randint(0, len(clean_seq) - 1)
    
    original_aa = clean_seq[mutation_position]
    aa_groups = find_groups(original_aa, groups)
    
    if aa_groups:
        selected_group = random.choice(aa_groups)
        
        if selected_group == "delete":
            mutated_aa = "-"
        elif selected_group == "insert":
            additional_aa = random.choice(groups["all"])
            mutated_seq = clean_seq[:mutation_position + 1] + additional_aa + clean_seq[mutation_position + 1:]
            return clean_sequence(mutated_seq)
        else:
            possible_mutations = [aa for aa in groups[selected_group] if aa != original_aa]
            mutated_aa = random.choice(possible_mutations) if possible_mutations else original_aa
    else:
        mutated_aa = original_aa
        
    mutated_seq = clean_seq[:mutation_position] + mutated_aa + clean_seq[mutation_position + 1:]
    
    return clean_sequence(mutated_seq)

def mutate(
    parent_seq: str, mutation_rate: float, num_trials: int = 10, mut_prob: float = 0.9
) -> Optional[str]:

    clean_seq = clean_sequence(parent_seq)
    
    if random.random() > mutation_rate:
        return clean_seq

    groups = {
        "positive": "RHK",
        "negative": "DE",
        "aromatic": "FWYH",
        "aliphatic": "VILM",
        "polar": "STHNQEDKR",
        "nonpolar": "VILFWYM",
        "DN_pair": "DN",
        "EQ_pair": "EQ",
        "small": "ASCTGP",
        "charged": "DERHK",
        "neutral": "ACFGHILMNPQSTVWY",
        "all": "ACDEFGHIKLMNPQRSTVWY",
        "insert": "ACDEFGHIKLMNPQRSTVWY",
        "delete": "ACDEFGHIKLMNPQRSTVWY"
    }
    new_seqs = []
    for _ in range(num_trials):
        try:
            proba = random.random()
            
            if proba <= 0.5:
                new_seqs.append(mutate_seq(clean_seq, groups))
                
                while True:
                    if random.random() <= mut_prob:
                        new_seqs.append(mutate_seq(random.choice([clean_seq] + new_seqs), groups))
                        mut_prob -= 0.1
                    else:
                        break
                
                if new_seqs is not None and len(new_seqs) > 0:
                    new_seqs = list(set(new_seqs))
                    return random.choice(new_seqs)
            else:
                new_seqs.append(find_homolog_seq(clean_seq))
                
                if new_seqs is not None and len(new_seqs) > 0:
                    new_seqs = list(set(new_seqs))
                    return random.choice(new_seqs)
        except:
            continue
    
    return None
