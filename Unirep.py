import sys
from tape import UniRepModel, TAPETokenizer
import time
import pandas as pd
import torch
import warnings
from Bio import SeqIO
from tqdm import tqdm

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
fasta_file = '/kaggle/input/gb-test-198-182-fasta/GB_test_198_182.fasta'
output_file = '/kaggle/working/unirep_embeddings_1900D.csv'

def read_fasta_file(filename):
    """Read FASTA file and return dictionary of sequences"""
    seq_dict = {}
    try:
        for record in SeqIO.parse(filename, "fasta"):
            seq_dict[record.id] = str(record.seq)
        return seq_dict
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}", file=sys.stderr)
        return None

def UniRep_Embed(input_seq):
    """Generate 1900D UniRep embeddings with 'Sequence ID' as first column"""
    T0 = time.time()
    embeddings_data = []
    sequence_ids = []
    print("\nGenerating UniRep Embeddings (1900D)...")

    try:
        model = UniRepModel.from_pretrained('babbler-1900')
        model = model.to(DEVICE)
        tokenizer = TAPETokenizer(vocab='unirep')
        
        for seq_id, sequence in tqdm(input_seq.items(), desc="Processing sequences"):
            if not sequence:
                print(f'WARNING: Skipping empty sequence {seq_id}', file=sys.stderr)
                continue
                
            with torch.no_grad():
                token_ids = torch.tensor([tokenizer.encode(sequence)]).to(DEVICE)
                output = model(token_ids)
                embedding = output[0].mean(dim=1).cpu().numpy()[0]  # 1900D vector
                
                sequence_ids.append(seq_id)
                embeddings_data.append(embedding)
        
        # Create DataFrame with 'Sequence ID' as first column
        df = pd.DataFrame(embeddings_data, 
                         columns=[f"UniRep_F{i+1}" for i in range(1900)])
        df.insert(0, 'Sequence ID', sequence_ids)  # Add ID column at position 0
        
        print(f"Completed in {(time.time()-T0)/60:.2f} minutes")
        return df
        
    except Exception as e:
        print(f"Error in embedding generation: {str(e)}", file=sys.stderr)
        return None

def process_fasta_and_save(fasta_file, output_file):
    """Process sequences and save embeddings with proper headers"""
    print(f"\n{'='*50}\nProcessing FASTA file: {fasta_file}")
    sequences = read_fasta_file(fasta_file)
    
    if not sequences:
        print(f"ERROR: No valid sequences found in {fasta_file}", file=sys.stderr)
        return
    
    embeddings = UniRep_Embed(sequences)

    if embeddings is not None:
        embeddings.to_csv(output_file, index=False)  # No duplicate index
        print(f"\nSaved embeddings to {output_file}")
        print(f"Output shape: {embeddings.shape} (rows x columns)")
    else:
        print("ERROR: Embedding generation failed.", file=sys.stderr)

if __name__ == "__main__":
    process_fasta_and_save(fasta_file, output_file)
    print("\nDone! Output columns: 'Sequence ID' + 1900 UniRep features.")