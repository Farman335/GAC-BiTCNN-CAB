import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO

# ✅ Use a publicly available ProGen2 model
model_name = "hugohrban/progen2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

def get_embedding(sequence):
    # Set max_length to avoid the truncation warning
    tokens = tokenizer(sequence, return_tensors='pt', truncation=True, padding=True, max_length=1024)
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1].squeeze(0)

    pooled = (hidden_states * attention_mask.squeeze(0).unsqueeze(-1)).sum(0) / attention_mask.sum()
    return pooled.cpu().numpy()

def fasta_to_csv(fasta_file, output_csv):
    headers, embeddings = [], []
    for rec in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Extracting ProGen2 embeddings"):
        seq = str(rec.seq).replace(" ", "").upper()
        try:
            emb = get_embedding(seq)
            headers.append(rec.id)
            embeddings.append(emb)
        except Exception as e:
            print(f"Error {rec.id}: {e}")

    # Create DataFrame with proper column names
    df = pd.DataFrame(embeddings)
    df.columns = [f"ProGen2-medium-F{i+1}" for i in range(df.shape[1])]  # Updated column names
    df.insert(0, "Sequence_ID", headers)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved embeddings to {output_csv}")

if __name__ == "__main__":
    fasta = "/kaggle/input/gb-test-198-182-fasta/GB_test_198_182.fasta"  # adjust path
    fasta_to_csv(fasta, "progen2-medium_embeddings_test.csv")