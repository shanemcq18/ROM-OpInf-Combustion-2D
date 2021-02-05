# rank_experiments.py

import numpy as np
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils



def main(k, maxr=100):
    ranks = []
    rs = np.arange(2,maxr+1)
    ds = 2 + rs + rs*(1 + rs)//2
    X_, Xdot_, t = utils.load_projected_data(k, max(rs))
    Us = config.U(t).reshape((1,-1))
    for r in rs:
        print(f"\rr = {r:d}", end='', flush=True)
        rom = roi.InferredContinuousROM(config.MODELFORM)
        D = rom._assemble_data_matrix(X_[:r,:], Us)
        ranks.append(np.linalg.matrix_rank(D))
    ranks = np.array(ranks)

    fig, ax = plt.subplots(1, 1)
    # ax.plot(ds, rs, 'C3-', lw=1, alpha=.5, label=r"$r$")
    ax.plot(ds, ds, 'C0-', lw=1, label=r"$d(r,m)$")
    ax.plot(ds, np.array(ranks), 'C3-',
             label=r"$\textrm{rank}(\mathbf{D})$")
    ax.set_xlim(0, ds[-1])
    ax.set_ylim(0, ds[-1])
    ax.set_xlabel(r"Data dimension $d(r,m)$")
    ax.set_ylabel(r"$\textrm{rank}(\mathbf{D})$")
    ax.set_title(fr"Rank deficiency of $\mathbf{{D}}$, $k = {k}$")        
    ax.legend(loc="lower right")

    ax_t = ax.secondary_xaxis("top")
    ax_t.set_xticks(ds[::4])
    ax_t.set_xticklabels(rs[::4])
    ax_t.set_xlabel(r"ROM size $r$")

    plt.show()
