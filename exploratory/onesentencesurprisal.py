from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch, math

text = "He placed a stack of papers next to the computer as the coffee machine beeped."

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

enc = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**enc, labels=enc["input_ids"])
    # Cross-entropy in nats/token:
    loss_nats = out.loss.item()
    # Convert to bits/token and perplexity:
    bits_per_token = loss_nats / math.log(2)
    ppl = math.exp(loss_nats)

# Total bits for the whole sequence (approx):
num_tokens = enc["input_ids"].shape[1]
total_bits = bits_per_token * num_tokens
prob = 2 ** (-total_bits)

print({
    "tokens": num_tokens,
    "bits_per_token": bits_per_token,
    "perplexity": ppl,
    "total_bits": total_bits,
    "approx_sentence_probability": prob
})
