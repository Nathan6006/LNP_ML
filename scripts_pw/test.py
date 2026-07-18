import torch, numpy as np
from transformers import AutoModel, AutoTokenizer
from config import BASE_MODEL

FT = "../finetuned_chemberta"
smiles = ["CCCCCC", "CCCCCCN(C)C", "O=C(OCC)CCN(CC)CC", "CN(C)CCOC(=O)CCCCCCCC", "CCCCCCCCCCCCOC(=O)CN(C)C"]

def emb(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).eval()   # watch the load warnings printed here
    enc = tok(smiles, padding=True, truncation=True, max_length=384, return_tensors="pt")
    with torch.no_grad():
        h = m(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)).numpy()

# compare the raw trunk weights too, not just embeddings
def trunk_sig(name):
    m = AutoModel.from_pretrained(name)
    return float(sum(p.abs().sum() for p in m.parameters()))


print("base trunk sig:", trunk_sig(BASE_MODEL))
print("ft   trunk sig:", trunk_sig(FT))
eb, ef = emb(BASE_MODEL), emb(FT)
print("max abs embedding diff:", np.abs(eb-ef).max())