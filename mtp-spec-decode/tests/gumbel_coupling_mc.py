"""Monte Carlo: standard rejection sampling with draft x=argmax(log q+G) and, on
rejection, resample y=argmax(log r+G') where G' is either the SAME noise as the
draft (what this image does) or fresh noise. Reports TV(output, p)."""
import numpy as np
rng = np.random.default_rng(0)
def run(p, q, shared, n=2_000_000):
    V = len(p); G = rng.gumbel(size=(n, V))
    x = np.argmax(np.log(q) + G, 1)
    acc = rng.random(n) < np.minimum(1, p[x] / q[x])
    r = np.maximum(p - q, 0); r = r / r.sum()
    G2 = G if shared else rng.gumbel(size=(n, V))
    with np.errstate(divide="ignore"):
        y = np.argmax(np.log(r) + G2, 1)
    out = np.where(acc, x, y)
    emp = np.bincount(out, minlength=V) / n
    return 0.5 * np.abs(emp - p).sum(), acc.mean()
cases = {
 "drafter close to target": (np.array([.5,.3,.15,.05]), np.array([.45,.3,.2,.05])),
 "drafter over-dispersed": (np.array([.7,.2,.08,.02]), np.array([.4,.3,.2,.1])),
 "drafter wrong argmax":   (np.array([.4,.35,.2,.05]), np.array([.2,.5,.2,.1])),
}
for name,(p,q) in cases.items():
    tv_s, a = run(p,q,True); tv_f, _ = run(p,q,False)
    print(f"{name:26s} accept={a:.3f}  TV(out,p) shared-noise={tv_s:.4f}  fresh-noise={tv_f:.4f}")
print("--- residual with >=2 tokens and very different q on them ---")
cases2 = {
 "p=(.1,.45,.45) q=(.8,.19,.01)": (np.array([.1,.45,.45]), np.array([.8,.19,.01])),
 "p=(.2,.4,.4)  q=(.6,.3,.1)":   (np.array([.2,.4,.4]), np.array([.6,.3,.1])),
 "p=(.05,.5,.3,.15) q=(.7,.25,.04,.01)": (np.array([.05,.5,.3,.15]), np.array([.7,.25,.04,.01])),
}
for name,(p,q) in cases2.items():
    tv_s, a = run(p,q,True,4_000_000); tv_f, _ = run(p,q,False,4_000_000)
    print(f"{name:38s} accept={a:.3f}  TV shared={tv_s:.4f}  fresh={tv_f:.4f}")
