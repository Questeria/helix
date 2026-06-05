#!/usr/bin/env python3
# FENCED OFFLINE ORACLE (Decision D1) for the Helix v1.0 capstone. A numpy reference
# (no torch) for the SAME tiny 2-layer pre-norm transformer that train_transformer.c
# trains on the GPU. Reads the C harness's init_weights.bin, runs the identical forward,
# backward, and Adam, and compares its loss curve to the GPU's loss_curve.csv for the
# within-2% capstone. NEVER imported by Helix training code -- an independent audit
# reference only. numpy is a numeric library, not an AI API.
#
# Usage: python3 oracle_train.py              (self-check the backward, train K=500, compare)
import numpy as np, sys, os

# Dims default to the v1.0 capstone (V=32,D=16,S=16,H=64,NL=2) so the existing v1.0
# capstone audit is byte-for-byte unchanged. The M6 scale-up re-train overrides them via
# env (HX_V/HX_D/HX_S/HX_H/HX_NL) to a representative size where the tiled+Tensor-Core
# kernels are valid (every matmul axis a multiple of 64) -- the SAME transformer math at a
# larger scale, so the 2% loss-parity check stays a real, identical-math correctness gate.
V  = int(os.environ.get("HX_V",  32))
D  = int(os.environ.get("HX_D",  16))
S  = int(os.environ.get("HX_S",  16))
H  = int(os.environ.get("HX_H",  64))
NL = int(os.environ.get("HX_NL", 2))
B1, B2, LR, EPS = 0.9, 0.999, 1e-3, 1e-8
K = int(os.environ.get("HX_K", 500))
SCALE = 1.0 / np.sqrt(float(D))   # attention scale 1/sqrt(d); 0.25 at d=16, 0.125 at d=64

def layernorm(x, g, b):
    mu = x.mean(1, keepdims=True); var = ((x - mu) ** 2).mean(1, keepdims=True); ist = 1.0 / np.sqrt(var)
    xhat = (x - mu) * ist
    return xhat * g + b, xhat, ist

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978846 * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    inn = 0.7978846 * (x + 0.044715 * x ** 3); th = np.tanh(inn); idd = 0.7978846 * (1.0 + 0.134145 * x * x)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * idd

def softmax(x):
    m = x.max(1, keepdims=True); e = np.exp(x - m); return e / e.sum(1, keepdims=True)

def read_weights(path="init_weights.bin"):
    w = np.fromfile(path, dtype="<f4").astype(np.float64); o = [0]
    def take(n, shape):
        a = w[o[0]:o[0] + n].reshape(shape).copy(); o[0] += n; return a
    W = {}
    for L in range(NL):
        W[f"Wq{L}"] = take(D * D, (D, D)); W[f"Wk{L}"] = take(D * D, (D, D)); W[f"Wv{L}"] = take(D * D, (D, D)); W[f"Wo{L}"] = take(D * D, (D, D))
        W[f"LN1g{L}"] = take(D, (D,)); W[f"LN1b{L}"] = take(D, (D,)); W[f"LN2g{L}"] = take(D, (D,)); W[f"LN2b{L}"] = take(D, (D,))
        W[f"W1{L}"] = take(D * H, (D, H)); W[f"W2{L}"] = take(H * D, (H, D))
    W["LNfg"] = take(D, (D,)); W["LNfb"] = take(D, (D,)); W["Wlm"] = take(D * V, (D, V))
    assert o[0] == w.size
    return W

x_in = np.zeros((S, D))
for s in range(S): x_in[s, s % D] = 1.0
tgt = np.array([(s + 1) % S for s in range(S)])

def forward(W, save=False):
    acts = {}; x = x_in.copy()
    for L in range(NL):
        xn1, xh1, ist1 = layernorm(x, W[f"LN1g{L}"], W[f"LN1b{L}"])
        Q = xn1 @ W[f"Wq{L}"]; Kk = xn1 @ W[f"Wk{L}"]; Vv = xn1 @ W[f"Wv{L}"]
        sc = SCALE * (Q @ Kk.T); at = softmax(sc); ao = at @ Vv; proj = ao @ W[f"Wo{L}"]; h1 = x + proj
        xn2, xh2, ist2 = layernorm(h1, W[f"LN2g{L}"], W[f"LN2b{L}"])
        a = xn2 @ W[f"W1{L}"]; g = gelu(a); m = g @ W[f"W2{L}"]; h2 = h1 + m
        if save: acts[L] = dict(x=x, xn1=xn1, xh1=xh1, ist1=ist1, Q=Q, K=Kk, Vv=Vv, at=at, ao=ao, h1=h1, xn2=xn2, xh2=xh2, ist2=ist2, a=a, g=g)
        x = h2
    xf, xhf, istf = layernorm(x, W["LNfg"], W["LNfb"])
    logits = xf @ W["Wlm"]
    if save: acts.update(xf=xf, xhf=xhf, istf=istf, logits=logits)
    return logits, acts

def loss_of(logits):
    p = softmax(logits); return float(-np.log(p[np.arange(S), tgt]).sum())  # SUM CE (matches the C)

def ln_bwd(dy, xhat, ist, gamma):
    dxhat = dy * gamma
    dx = ist * (dxhat - dxhat.mean(1, keepdims=True) - xhat * (dxhat * xhat).mean(1, keepdims=True))
    return dx, (dy * xhat).sum(0), dy.sum(0)

def backward(W, acts):
    G = {}; logits = acts["logits"]; p = softmax(logits)
    dlog = p.copy(); dlog[np.arange(S), tgt] -= 1.0
    G["Wlm"] = acts["xf"].T @ dlog
    dxf = dlog @ W["Wlm"].T
    dh, G["LNfg"], G["LNfb"] = ln_bwd(dxf, acts["xhf"], acts["istf"], W["LNfg"])
    for L in reversed(range(NL)):
        ac = acts[L]; dm = dh
        G[f"W2{L}"] = ac["g"].T @ dm
        dg = dm @ W[f"W2{L}"].T
        da = dg * gelu_grad(ac["a"])
        G[f"W1{L}"] = ac["xn2"].T @ da
        dxn2 = da @ W[f"W1{L}"].T
        dxl2, G[f"LN2g{L}"], G[f"LN2b{L}"] = ln_bwd(dxn2, ac["xh2"], ac["ist2"], W[f"LN2g{L}"])
        dh1 = dxl2 + dm
        G[f"Wo{L}"] = ac["ao"].T @ dh1
        dao = dh1 @ W[f"Wo{L}"].T
        dVv = ac["at"].T @ dao
        dattn = dao @ ac["Vv"].T
        dsc = ac["at"] * (dattn - (dattn * ac["at"]).sum(1, keepdims=True))
        dQ = SCALE * (dsc @ ac["K"]); dK = SCALE * (dsc.T @ ac["Q"])
        G[f"Wq{L}"] = ac["xn1"].T @ dQ; G[f"Wk{L}"] = ac["xn1"].T @ dK; G[f"Wv{L}"] = ac["xn1"].T @ dVv
        dxn1 = dQ @ W[f"Wq{L}"].T + dK @ W[f"Wk{L}"].T + dVv @ W[f"Wv{L}"].T
        dxl1, G[f"LN1g{L}"], G[f"LN1b{L}"] = ln_bwd(dxn1, ac["xh1"], ac["ist1"], W[f"LN1g{L}"])
        dh = dxl1 + dh1
    return G

def selfcheck(W):
    G = backward(W, forward(W, save=True)[1]); h = 1e-4; bad = 0
    for kk in ["Wlm", "W21", "Wo1", "Wq0", "W10", "LN2g1"]:
        Wt = W[kk]; Gt = G[kk]
        for j in range(min(3, Wt.size)):
            idx = tuple(np.unravel_index((j * 13 + 5) % Wt.size, Wt.shape))
            s = Wt[idx]; Wt[idx] = s + h; lp = loss_of(forward(W)[0]); Wt[idx] = s - h; lm = loss_of(forward(W)[0]); Wt[idx] = s
            fd = (lp - lm) / (2 * h); gv = Gt[idx]; e = abs(gv - fd)
            if e > 1e-3 and e > 0.05 * abs(fd): print(f"  selfcheck {kk}{idx} grad {gv:.6f} fd {fd:.6f} FAIL"); bad += 1
    print(f"oracle backward self-check: {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0

def train(W):
    m_ = {k: np.zeros_like(v) for k, v in W.items()}; v_ = {k: np.zeros_like(v) for k, v in W.items()}
    curve = {}
    for t in range(1, K + 1):
        logits, acts = forward(W, save=True); L = loss_of(logits)
        if t == 1 or t % 25 == 0 or t == K: curve[t] = L
        G = backward(W, acts); bc1 = 1.0 / (1.0 - B1 ** t); bc2 = 1.0 / (1.0 - B2 ** t)
        for k in W:
            g = G[k]; m_[k] = B1 * m_[k] + (1 - B1) * g; v_[k] = B2 * v_[k] + (1 - B2) * g * g
            W[k] = W[k] - LR * (m_[k] * bc1) / np.sqrt(v_[k] * bc2 + EPS)
    curve[K + 1] = loss_of(forward(W)[0])
    return curve

if __name__ == "__main__":
    W = read_weights()
    logits, _ = forward(W, save=True)
    print(f"oracle step0 loss = {loss_of(logits):.6f}")
    ok = selfcheck(read_weights())
    # v1.3 audit-remediation A5: a failed backward self-check must FAIL CLOSED
    # (previously `ok` was computed but never gated -> the oracle continued and
    # exited 0 even on a broken gradient, masking the failure from capstone_audit.sh).
    if not ok:
        print("oracle backward self-check FAILED -- aborting (analytic backprop disagrees with finite-diff)")
        sys.exit(1)
    curve = train(read_weights())
    with open("oracle_curve.csv", "w") as f:
        for t in sorted(curve): f.write(f"{t},{curve[t]:.8f}\n")
    print(f"oracle train K={K}: start {curve[1]:.6f} -> final {curve[K+1]:.6f}")
    if os.path.exists("loss_curve.csv"):
        gpu = {}
        for line in open("loss_curve.csv"):
            t, l = line.strip().split(","); gpu[int(t)] = float(l)
        worst = 0.0; worst_t = 0
        for t in sorted(curve):
            if t in gpu:
                rel = abs(gpu[t] - curve[t]) / (abs(curve[t]) + 1e-8)
                if rel > worst: worst = rel; worst_t = t
        print(f"capstone compare: worst relative diff = {worst*100:.4f}% at step {worst_t} -> {'CAPSTONE PASS within 2%' if worst < 0.02 else 'FAIL >2%'}")
