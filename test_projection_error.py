
import os
import h5py
import logging
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc





def check_projection_training_error(trainsize, rs):

    # Load, lift, and scale GEMS data.
    try:
        X, t, scales = utils.load_scaled_data(trainsize)
    except utils.DataNotFoundError:
        data, t = utils.load_gems_data(cols=trainsize)
        with utils.timed_block("preprocessing GEMS data"):
            data_lifted = dproc.lift(data)
            del data
            X, scales = dproc.scale(data_lifted)
            del data_lifted

    # Load full array of singular values.
    svdval_path = os.path.join(config.BASE_FOLDER,
                               config.TRNFMT(trainsize), "svdvals.h5")
    with h5py.File(svdval_path, 'r') as hf:
        svdvals = hf["svdvals"][:]

    # Check the projection error and the singular value residual.
    if np.isscalar(rs):
        rs = [rs]
    for r in rs:
        Vr, scales = utils.load_basis(trainsize, r)
        proj_residual = X - (Vr @ (Vr.T @ X))
        e_proj1 = np.sum(proj_residual**2) / np.sum(X**2)
        e_proj2 = 1 - np.sum(svdvals[:r]**2) / np.sum(svdvals**2)
        if np.allclose(e_proj1, e_proj2):
            print(f"Projection errors match, r = {r}")
        else:
            print(f"PROJECTION ERRORS INCONSISTENT, r = {r}")
            print(f"||(I - VV.T)X||_F^2 / ||:", e_proj1)
            print(f"1 - sum(svdvals[:r]^2)/sum(svdvals^2)", e_proj2)


def check_projection_prediction_error(trainsize, rs, subset=True):
    # Load the two bases to compare.
    assert len(rs) == 2, "need 2 basis sizes to compare"
    V1, scales = utils.load_basis(trainsize, rs[0])
    V2, _ = utils.load_basis(trainsize, rs[1])
    
    # Load the snapshots to test.
    if subset:
        t = list(range(trainsize-1000, trainsize+1050, 50))
        data, _ = utils.load_gems_data(cols=t)
    else:
        data, t = utils.load_gems_data()
    with utils.timed_block("preprocessing GEMS data"):
        data_lifted = dproc.lift(data)
        del data
        X, scales = dproc.scale(data_lifted)
        del data_lifted

    # Check the projection error and the singular value residual.
    with utils.timed_block("computing projection errors"):
        denom = la.norm(X, axis=0)
        X1 = X - (V1 @ (V1.T @ X))
        X2 = X1 - (V2 @ (V2.T @ X1))
    plt.plot(t, la.norm(X1, axis=0)/denom, label=fr"$r = {rs[0]}$")
    plt.plot(t, la.norm(X2, axis=0)/denom, label=fr"$r = {rs[1]}$")
    plt.axvline(trainsize, color='k')
    plt.xlabel("time index")
    plt.legend(loc="lower right")

    # Save the figure.
    utils.save_figure("projection_check.pdf")
    return


def single_basis_singular_values(trainsize, var="T"):
    assert var in config.GEMS_VARIABLES and var in config.ROM_VARIABLES
    start = config.GEMS_VARIABLES.index(var)*config.DOF
    _, scales = utils.load_basis(trainsize, 10)

    # Load and scale all GEMS temperature snapshots (no lifting needed).
    data, t = utils.load_gems_data(rows=slice(start, start+config.DOF),
                                   cols=trainsize)
    with utils.timed_block(f"Scaling GEMS {var} data"):
        data, _ = dproc.scale(data, scales, variables=var)

    # Compute all `trainsize` singular values of the data set.
    with utils.timed_block(f"Computing dense SVD for {var} data"):
        svdvals = la.svdvals(data)

    svdfile = os.path.join(config.BASE_FOLDER,
                           config.TRNFMT(trainsize), f"svdvals.h5")
    with utils.timed_block(f"Saving svdvals to {svdfile}"):
        with h5py.File(svdfile, 'a') as hf:
            hf.create_dataset(var, data=svdvals)
    return


def plot_singular_values():
    fig, axes = plt.subplots(2, 2, figsize=(12,6), sharex=True, sharey=True)
    
    for trainsize, ax in zip([5000, 10000, 20000, 30000], axes.flat):
        svdfile = os.path.join(config.BASE_FOLDER,
                               config.TRNFMT(trainsize), f"svdvals.h5")
        with h5py.File(svdfile, 'r') as hf:
            allvals = hf["svdvals"][:]
            Tvals = hf["T"][:]
            pvals = hf["p"][:]
            vxvals = hf["vx"][:]

        j = np.arange(1, allvals.size + 1)
        ax.semilogy(j, allvals/allvals[0], color="C0", lw=1, label=r"All variables")
        ax.semilogy(j,   Tvals/  Tvals[0], color="C1", lw=1, label=r"$T$ only")
        ax.semilogy(j,   pvals/  pvals[0], color="C2", lw=1, label=r"$p$ only")
        ax.semilogy(j,  vxvals/ vxvals[0], color="C3", lw=1, label=r"$v_{x}$ only")
        ax.set_title(fr"$k = {trainsize}$")
        ax.set_xlim(0, 200)
        ax.set_ylim(bottom=1e-5)

    # Make legend centered below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    leg = axes[0,0].legend(ncol=4, fontsize=12, loc="lower center",
                           bbox_to_anchor=(.5, 0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linestyle('-')
        line.set_linewidth(5)
    
    utils.save_figure("svdval_comparison.pdf")


def single_basis_projection_error(trainsize, rs, var="T"):
    assert var in config.GEMS_VARIABLES and var in config.ROM_VARIABLES
    start = config.GEMS_VARIABLES.index(var)*config.DOF
    _, scales = utils.load_basis(trainsize, rs[0])

    # Load and scale all GEMS temperature snapshots (no lifting needed).
    data, t = utils.load_gems_data(rows=slice(start, start+config.DOF))
    with utils.timed_block("Scaling GEMS temperature data"):
        data, _ = dproc.scale(data, scales, variables=var)

    data_train, data_test = np.split(data, [trainsize], axis=1)
    assert data_train.shape == (config.DOF, trainsize)
    assert data_test.shape[0] == config.DOF
    rmax = max(rs)
    with utils.timed_block(f"Computing rank-{rmax} POD basis"):
        V, _ = roi.pre.pod_basis(data_train, r=rmax,
                                 mode="randomized", n_iter=15, random_state=42)
    
    # Initialize a figure.
    fig, ax = plt.subplots(1, 1)

    # Compute and plot projection errors.
    denom = np.abs(data).max(axis=0)
    for r in rs:
        Vr = V[:,:r]
        with utils.timed_block(f"Computing rank-{r} projection error"):
            projection = Vr @ (Vr.T @ data)
            error = np.abs(data - projection) / denom
            error_mean = np.mean(error, axis=0)
            error_std = np.std(error, axis=0)
            ax.plot(t, error_mean, '-', lw=.5, label=fr"$r = {r}$")
    ax.axvline(t[trainsize], color='k', lw=1)

    # Format the figure.
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel(config.VARTITLES[var])
    ax.set_title("Projection Error for single-variable basis")

    # Make legend centered below main plot.
    fig.tight_layout(rect=[0, .1, 1, 1])
    leg = ax.legend(ncol=3, fontsize=14, loc="lower center",
                    bbox_to_anchor=(.5, 0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linestyle('-')
        line.set_linewidth(5)



    utils.save_figure(f"{var}basis_projection_error.pdf")


if __name__ == "__main__":
    utils.reset_logger()
