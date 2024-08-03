#%%
words = open('names.txt', 'r').read().splitlines()
# %%
words[:10]
# %%
len(words)
# %%
min(len(w) for w in words)
# %%
max(len(w) for w in words)

# %%
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# %%
b
# %%
sorted(b.items(), key=lambda x: -x[1])
# %%
import torch
# %%
N = torch.zeros((27 ,27), dtype=torch.int32)
# %%
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
stoi
# %%
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xi1 = stoi[ch1]
        xi2 = stoi[ch2]
        N[xi1, xi2] += 1
# %%
import matplotlib.pyplot as plt
# %%

# %%
itos = {i:s for s, i in stoi.items()}
itos
# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
# %%
N[0]
# %%
p = N[0].float()
p /= p.sum()
p
# %%
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]
#%%
P = (N + 1).float()
P /= P.sum(dim=1, keepdim=True)
P
# %%
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
# %%
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xi1 = stoi[ch1]
        xi2 = stoi[ch2]
        prob = P[xi1, xi2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print(f'{log_likelihood.item():.4f}')
# %%
nll = -log_likelihood
print(f'{nll.item():.4f}')
print(f'{nll.item()/n:.4f}')
# %%
# create the tranining set of bigrams (x, y)
xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xi1 = stoi[ch1]
        xi2 = stoi[ch2]
        xs.append(xi1)
        ys.append(xi2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
# %%
import torch.nn.functional as F
# %%
xenc = F.one_hot(xs, num_classes=27).float()
# %%
xenc.shape
# %%
W = torch.randn((27, 27))
(xenc @ W).shape
# %%
logits = xenc @ W # log-counts
# %%
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
# %%
probs[0].shape
# %%
nlls = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    print("------")
    print(f"bigram example {i + 1}: {itos[x]}{itos[y]} (indexes {x}, {y})")
    print("input to the neural net:", x)
    print("output probabilities from neural net:", probs[i])
    print("label (actual next character):", y)
    p = probs[i, y]
    print("probability assigned by the net to the correct character", p.item())
    logp = torch.log(p)
    print("log likelihood:", logp.item())
    nlls[i] = -logp
print("=======")
print("average negative log likelihood", nlls.mean().item())
# %%
