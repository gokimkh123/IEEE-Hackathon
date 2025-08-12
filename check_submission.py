import argparse, pickle, numpy as np
from pathlib import Path

def main(a):
    d = pickle.load(open(a.pkl, "rb"))
    total = 0; ok_shape = ok_bin = 0; nonempty = 0
    for part in sorted(d):
        for it in d[part]:
            total += 1
            s = it.get("streak_label"); p = it.get("spot_label")
            cond_shape = isinstance(s,np.ndarray) and isinstance(p,np.ndarray) and s.shape==(139,250) and p.shape==(139,250)
            cond_bin = np.isin(s, [0,1]).all() and np.isin(p, [0,1]).all()
            ok_shape += int(cond_shape); ok_bin += int(cond_bin)
            nonempty += int((s.sum()>0) or (p.sum()>0))
    print(f"Total layers: {total}")
    print(f"Shape ok (139x250): {ok_shape}/{total}")
    print(f"Binary ok (0/1): {ok_bin}/{total}")
    print(f"Has any positives: {nonempty}/{total} layers")
    print("OK!" if (ok_shape==total and ok_bin==total) else "Found issues.")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="NIST_Task1.pkl")
    a = ap.parse_args(); main(a)
