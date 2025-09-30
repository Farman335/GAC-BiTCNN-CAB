import sys
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import numpy as np
import pandas as pd
import torch
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# ✅ Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Using CPU - Consider GPU for faster processing")

# ✅ Load ProtBERT model (with error handling)
try:
    print("Loading ProtBERT model and tokenizer...")
    model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    sys.exit(1)

# ✅ File paths
fasta_file = 'C:/Users/Mujtaba/Desktop/UniproLcad-main/data/test/sequences.fasta'
output_file = './protbert_features.csv'

# ✅ Parse FASTA with validation
print(f"\nProcessing FASTA file: {fasta_file}")
try:
    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_file, 'fasta')}
    if not sequences:
        print("❌ Error: No sequences found in FASTA file")
        sys.exit(1)
except Exception as e:
    print(f"❌ FASTA parsing error: {str(e)}")
    sys.exit(1)

# ✅ Feature extraction function
def extract_protbert_features(sequence_dict, batch_size=4):
    """Extract ProtBERT features with memory optimization"""
    all_features = []
    sequence_ids = []
    processed_seqs = []
    
    # Pre-process sequences
    for seq_id, seq in sequence_dict.items():
        if not seq:
            print(f"⚠️ Skipping empty sequence: {seq_id}")
            continue
        processed_seqs.append((" ".join(seq), seq_id))
    
    # Batch processing
    for i in tqdm(range(0, len(processed_seqs), batch_size), 
                    desc="Extracting features", unit="batch"):
        batch = processed_seqs[i:i+batch_size]
        seqs, ids = zip(*batch)
        
        try:
            # Tokenize with automatic padding
            inputs = tokenizer(
                seqs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=1024  # ProtBERT max length
            ).to(DEVICE)
            
            # Forward pass with mixed precision
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(**inputs)
            
            # Process embeddings (mean pooling)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            
            # Mean pooling with attention mask
            masked_embeddings = embeddings * attention_mask
            summed = torch.sum(masked_embeddings, dim=1)
            counts = torch.sum(attention_mask, dim=1)
            mean_pooled = (summed / counts).cpu().numpy()
            
            all_features.extend(mean_pooled)
            sequence_ids.extend(ids)
            
        except Exception as e:
            print(f"⚠️ Batch {i//batch_size} failed: {str(e)}")
            continue
        
        # Memory management
        if i % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return sequence_ids, np.array(all_features)

# ✅ Main processing
print("\nStarting feature extraction...")
protein_ids, features = extract_protbert_features(sequences)

# Create DataFrame
feature_df = pd.DataFrame(
    features,
    columns=[f"ProtBert_F{i+1}" for i in range(features.shape[1])]
)
feature_df.insert(0, "Sequence ID", protein_ids)

print(f"\n✅ Feature extraction complete!")
print(f"Shape: {feature_df.shape} (sequences × features)")

# Save results
print(f"\nSaving to {output_file}...")
feature_df.to_csv(output_file, index=False)
print("Done! 🎉")