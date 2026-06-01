#!/usr/bin/env python3
# FENCED OFFLINE ORACLE (Decision D1) for the Helix v1.0 capstone. A numpy reference
# for the SAME tiny 2-layer pre-norm transformer that train_transformer.c trains on the
# GPU. Reads the C harness's init_weights.bin (identical weights), runs the identical
# forward (and, later, backward + Adam), and reports the loss for the within-2% capstone
# comparison. This script is NEVER imported by or called from any Helix training code --
# it is an independent audit reference only. numpy is a numeric library, not an AI API.
#
# Usage: python3 oracle_train.py            (reads ./init_weights.bin, prints step0 loss)
#        python3 oracle_train.py <K>        (also runs K Adam steps -- Stage E, TODO)
import numpy as np, sys

V, D, S, H, NL = 32, 16, 16, 64, 2

def layernorm(x, g, b):           # over the last dim (size D); NO eps (matches the GPU kernel)
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var) * g + b

def gelu(x):                      # tanh approximation, matches gpu_gelu
    return 0.5 * x * (1.0 + np.tanh(0.7978846 * (x + 0.044715 * x ** 3)))

def softmax(x):                   # row-wise
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)

# ---- read init_weights.bin in the exact train_transformer.c order ----
w = np.fromfile("init_weights.bin", dtype="<f4").astype(np.float64)
_o = 0
def take(n, shape):
    global _o
    a = w[_o:_o + n].reshape(shape).copy(); _o += n; return a
W = {}
for L in range(NL):
    W[f"Wq{L}"] = take(D * D, (D, D)); W[f"Wk{L}"] = take(D * D, (D, D))
    W[f"Wv{L}"] = take(D * D, (D, D)); W[f"Wo{L}"] = take(D * D, (D, D))
    W[f"LN1g{L}"] = take(D, (D,)); W[f"LN1b{L}"] = take(D, (D,))
    W[f"LN2g{L}"] = take(D, (D,)); W[f"LN2b{L}"] = take(D, (D,))
    W[f"W1{L}"] = take(D * H, (D, H)); W[f"W2{L}"] = take(H * D, (H, D))
W["LNfg"] = take(D, (D,)); W["LNfb"] = take(D, (D,)); W["Wlm"] = take(D * V, (D, V))
assert _o == w.size, (_o, w.size)

# ---- inputs (identical to the C harness) ----
x_in = np.zeros((S, D));
for s in range(S): x_in[s, s % D] = 1.0
tgt = np.array([(s + 1) % S for s in range(S)])

def forward(W):
    x = x_in.copy()
    for L in range(NL):
        xn1 = layernorm(x, W[f"LN1g{L}"], W[f"LN1b{L}"])
        Q = xn1 @ W[f"Wq{L}"]; K = xn1 @ W[f"Wk{L}"]; Vv = xn1 @ W[f"Wv{L}"]
        scores = 0.25 * (Q @ K.T)
        attn = softmax(scores)
        ao = attn @ Vv
        proj = ao @ W[f"Wo{L}"]
        h1 = x + proj
        xn2 = layernorm(h1, W[f"LN2g{L}"], W[f"LN2b{L}"])
        a = xn2 @ W[f"W1{L}"]; g = gelu(a); m = g @ W[f"W2{L}"]
        x = h1 + m
    xf = layernorm(x, W["LNfg"], W["LNfb"])
    logits = xf @ W["Wlm"]
    return logits

def loss_of(logits):
    p = softmax(logits)
    return float(-np.log(p[np.arange(S), tgt]).mean())

logits = forward(W)
L0 = loss_of(logits)
print(f"oracle step0 loss = {L0:.6f}  logits[0,0]={logits[0,0]:.5f}")
