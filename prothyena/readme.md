source paper: https://www.biorxiv.org/content/10.1101/2024.01.18.576206v1

weights: https://huggingface.co/pandeyps/fela

protein language model on the hyena operator (not a complete implementation of the paper, as its just the base model and not finetune, hence thought to name it fela.)

- Architecture: long conv + MLP blocks, pre-norm, LM head
- Tokenizer: char level over `ACDEFGHIKLMNPQRSTVWYX`, `<pad>`=0, `<eos>`=22, `<unk>`=23
- Data: Pfam-A (filtered to 20–512 residues, standard alphabet only), ~9.5B tokens
- Training: 40k steps, batch 256, bf16, AdamW (wd 0.1), cosine LR 6e-4 → 6e-5


base model - 1.6M params (not finetuned)

#### Config

| Parameter | Value |
|---|---|
| d_model | 256 |
| n_layer | 2 |
| d_inner | 1024 |
| vocab_size | 32 |
| l_max | 514 |
| order | 2 |
| filter_order | 64 |
| short_filter_order | 3 |
| emb_dim | 5 |
| w | 10 |
| num_inner_mlps | 2 |
| residual_in_fp32 | true |

#### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("pandeyps/fela", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("pandeyps/fela", trust_remote_code=True)

ids = tok.encode("MSDKIIEYDETARRAIEAGVNTLADAV", return_tensors="pt")
gen = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7)
print(tok.decode(gen[0]))