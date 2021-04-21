# rank_experiments.py
"""Final project for CSE 382M."""

import os
import h5py
import logging
import numpy as np
import scipy.linalg as la
import scipy.spatial as sp
import matplotlib.pyplot as plt

import rom_operator_inference as opinf

import config
import utils
import data_processing as dproc
import step2a_transform as step2a


# Subroutines =================================================================

def weights_time_decay(time_domain, sigma=2):
    """Construct weights based on the time (smaller time, greater weight):

    w_j = σ^(t_j / t_{k-1}),  j = 0, 1, ..., k - 1 = trainsize - 1.

    Parameters
    ----------
    time_domain : (trainsize,) ndarray
        Time domain corresponding to the training snapshots.

    sigma : float > 1
        Base of exponential.

    Returns
    -------
    w : (trainsize,) ndarray
        Snapshot weights.
    """
    t = time_domain - time_domain[0]
    return sigma**(-t/t[-1])


def weights_gaussian(training_data, sigma=1, k=None, kernelize=True):
    """Construct weights based on the Gaussian kernel (spatial importance):

    K(xi, xj) = exp(-||xi - xj||^2 / 2σ^2)

    Parameters
    ----------
    training_data : (n,k) ndarray
        Training snapshots, pre-processed except for mean shifting.

    sigma : float > 0
        Gaussian kernel spread hyperparameter.

    k : int > 0 or None
        Dimension of random projection to approximate distances.

    kernelize : bool
        If True, apply the Gaussian kernel. If False, use squared Euclidean
        distances (no kernel).
    """
    # If k is given, randomly project the data to r dimensions.
    if k is not None:
        M = np.random.standard_normal((training_data.shape[0], k))
        X = (M.T @ training_data).T
    else:
        X = training_data.T
        k = 1

    # Calculate the kernel matrix and the resulting weights.
    distances = sp.distance.pdist(X, "sqeuclidean") / k
    if kernelize:
        distances = np.exp(-distances/(2*sigma**2))
    K = sp.distance.squareform(distances)
    return np.mean(K, axis=1)


def center_snapshots(data, weights=None):
    """Center and weight a snapshot matrix (NOT in place).

    Parameters
    ----------
    data : (n,k) ndarray
        Data to weight / shift (already non-dimensionalized).

    weights : (k,) ndarray or None
        Nonnegative weights corresponding to the snapshots.

    Returns
    -------
    data_centered : (n,k) ndarray
        Data matrix such that the weighted sum of columns is zero:
        sum_{j} data_centered[:,j]*weights[j] = 0.

    qbar : (n,) ndarray
        Weighted mean of snapshots.
    """
    if weights is None:
        qbar = np.mean(data, axis=1)                # Compute unweighted mean.
    else:
        weights /= np.sum(weights)                  # Normalize weights.
        qbar = (data @ weights)                     # Compute weighted mean.
    data_centered = data - qbar.reshape((-1,1))     # Shift by computed mean.
    return data_centered, qbar


def compute_basis(data, r, weights=None):
    """Compute the weighted (randomized) SVD up to r vectors.

    Parameters
    ----------
    data : (n,k) ndarray
        Data to take the SVD of (already weighted).

    r : int
        How many singular values/vectors to compute.

    weights : (k,) ndarray
        Nonnegative weights corresponding to the snapshots.

    Returns
    -------
    V : (n,r) ndarray
        Approximate first r left singular vectors of the data.

    svdvals : (r,) ndarray
        Approximate first r singular values of the data.
    """
    X = data if weights is None else data * weights
    return opinf.pre.pod_basis(X, r=r,
                               mode="randomized", n_iter=15, random_state=42)


def group_exists(filename, groupname):
    """Return True if the given group exists in the specified h5 file."""
    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as hf:
            return False
    with h5py.File(filename, 'r') as hf:
        return groupname in hf


# Main routines ===============================================================

def single_experiment(weight_type, σ, rmax,
                      time_domain, training_data, scales,
                      filename):
    """Generate and save data for a single weighted POD experiment."""
    # Get the specified type of weights.
    with utils.timed_block(f"Computing {weight_type} weights, σ={σ}"):
        if weight_type == "temporal":
            weights = weights_time_decay(time_domain, sigma=σ)
        elif weight_type == "spatial":
            weights = weights_gaussian(training_data, sigma=σ, k=1000)
        elif weight_type == "plain":
            weights = weights_gaussian(training_data, k=1000, kernelize=False)
        else:
            raise ValueError(f"invalid weight_type '{weight_type}'")

    # Get the associated POD basis and project the training data.
    with utils.timed_block("Centering snapshot matrix"):
        data_centered, qbar = center_snapshots(training_data, weights)
    with utils.timed_block(f"Computing {rmax:d}-component rSVD"):
        V, basis_svdvals = compute_basis(data_centered, rmax, weights)
    with utils.timed_block("Projecting data"):
        Q_ = V.T @ data_centered
        dt = time_domain[1] - time_domain[0]
        Us = config.U(time_domain).reshape((1, -1))
        Qdot_ = opinf.pre.xdot_uniform(Q_, dt, order=4)

    # Compute the condition number and the rank of the data matrix for each r.
    with utils.timed_block("Computing data matrix statistics"):
        rom = opinf.InferredContinuousROM(config.MODELFORM)
        ε = np.finfo(float).eps
        conds, ranks = [], []
        for r in range(2, rmax+1):
            print(f"\rr = {r:d}", end=' '*40, flush=True)
            D = rom._assemble_data_matrix(Q_[:r,:], Us)
            data_svdvals = la.svdvals(D)
            # Exclude svdval[0], which is very large due to input u(t).
            cutoff = data_svdvals[1] * max(D.shape) * ε
            conds.append(data_svdvals[0] / data_svdvals[-1])
            ranks.append(np.sum(data_svdvals > cutoff))

    # Save the data for later.
    with utils.timed_block(f"Saving experiment data for σ={σ}"):
        with h5py.File(filename, 'a') as hf:
            groupname = f"{weight_type}/sigma_{σ:02d}"
            if groupname in hf:
                print(f"OVERWRITING DATA for group {groupname}")
                del hf[groupname]
            group = hf.require_group(groupname)
            group.create_dataset("weights", data=weights)
            group.create_dataset("time", data=time_domain)
            group.create_dataset("scales", data=scales)
            group.create_dataset("mean_snapshot", data=qbar)
            group.create_dataset("basis", data=V)
            group.create_dataset("svdvals", data=basis_svdvals)
            group.create_dataset("projected_snapshots", data=Q_)
            group.create_dataset("projected_derivatives", data=Qdot_)
            group.create_dataset("dataconds", data=np.array(conds))
            group.create_dataset("dataranks", data=np.array(ranks))


def dqs_experiment(dqorder, rmax,
                   time_domain, training_data, scales, filename):
    """Compute difference quotients, then basis, THEN project."""

    # Get the associated POD basis and project the training data.
    with utils.timed_block("Centering snapshot matrix"):
        data_centered, qbar = center_snapshots(training_data, None)
    with utils.timed_block(f"Computing {dqorder}-order difference quotients"):
        dt = time_domain[1] - time_domain[0]
        time_derivatives = opinf.pre.xdot_uniform(data_centered, dt, dqorder)
    with utils.timed_block(f"Computing {rmax:d}-component rSVD"):
        V, basis_svdvals = compute_basis(np.hstack([data_centered,
                                                    time_derivatives]),
                                         rmax, weights=None)
    with utils.timed_block("Projecting data"):
        Q_ = V.T @ data_centered
        Qdot_ = V.T @ time_derivatives
        Us = config.U(time_domain).reshape((1, -1))

    # Compute the condition number and the rank of the data matrix for each r.
    with utils.timed_block("Computing data matrix statistics"):
        rom = opinf.InferredContinuousROM(config.MODELFORM)
        ε = np.finfo(float).eps
        conds, ranks = [], []
        for r in range(2, rmax+1):
            print(f"\rr = {r:d}", end=' '*40, flush=True)
            D = rom._assemble_data_matrix(Q_[:r,:], Us)
            data_svdvals = la.svdvals(D)
            # Exclude svdval[0], which is very large due to input u(t).
            cutoff = data_svdvals[1] * max(D.shape) * ε
            conds.append(data_svdvals[0] / data_svdvals[-1])
            ranks.append(np.sum(data_svdvals > cutoff))

    # Save the data for later.
    with utils.timed_block(f"Saving DQ experiment data with order {dqorder}"):
        with h5py.File(filename, 'a') as hf:
            groupname = f"DQs/order_{dqorder:d}"
            if groupname in hf:
                print(f"OVERWRITING DATA for group {groupname}")
                del hf[groupname]
            group = hf.require_group(groupname)
            group.create_dataset("time", data=time_domain)
            group.create_dataset("scales", data=scales)
            group.create_dataset("mean_snapshot", data=qbar)
            group.create_dataset("basis", data=V)
            group.create_dataset("svdvals", data=basis_svdvals)
            group.create_dataset("projected_snapshots", data=Q_)
            group.create_dataset("projected_derivatives", data=Qdot_)
            group.create_dataset("dataconds", data=np.array(conds))
            group.create_dataset("dataranks", data=np.array(ranks))


def main(k, rmax, sigmas, filename="experiment.h5"):
    """Generate experiment data for a group of sigmas.

    Parameters
    ----------
    k : int
        Number of training snapshots.

    rmax : int
        Maximum basis size to use.

    sigmas : list(float)
        The hyperparameters to use in the weight computations.
    """
    filename = os.path.join(config.BASE_FOLDER, config.TRNFMT(k), filename)

    # Get k lifted snapshots and scale the data.
    training_data, time_domain = step2a.load_and_lift_gems_data(k)
    with utils.timed_block(f"Scaling {k:d} lifted snapshots"):
        training_data, scales = dproc.scale(training_data)

    # Compute spatial weights based on squared Euclidean distances.
    if not group_exists(filename, "plain/sigma_0"):
        single_experiment("plain", 0, rmax,
                          time_domain, training_data, scales, filename)

    # For each choice of sigma, compute the weights.
    for σ in sigmas:
        for wtype in ["temporal", "spatial"]:
            if not group_exists(filename, f"{wtype}/sigma_{σ:02d}"):
                print(f"\nNew experiment with {wtype} weights, σ={σ}:")
                single_experiment(wtype, σ, rmax,
                                  time_domain, training_data, scales,
                                  filename)

    for dqorder in [2, 4, 6]:
        if not group_exists(filename, f"DQs/order_{dqorder:d}"):
            dqs_experiment(dqorder, rmax,
                           time_domain, training_data, scales, filename)


def plot_results(k, filename="experiment.h5", fontsize="large"):
    """Make lots of plots."""
    filename = os.path.join(config.BASE_FOLDER, config.TRNFMT(k), filename)
    plt.rc("axes", titlesize=fontsize, labelsize=fontsize)
    plt.rc("legend", fontsize=fontsize)

    def uniform_weight_style(line):
        line.set_linewidth(1)
        line.set_color('k')
        line.set_zorder(4)

    def plot_weights(ax, t, weights, name):
        """Plot weights as a function of time on the given axes."""
        line = ax.plot(t, weights / np.sum(weights), '-', lw=.5, label=name)[0]
        if name == "Uniform":
            uniform_weight_style(line)

    def plot_svdvals(ax, svdvals, name):
        """Plot singular values on the given axes."""
        j = np.arange(1, len(svdvals)+1)
        line = ax.semilogy(j, svdvals/svdvals[0], '-', lw=.5, label=name)[0]
        if name == "Uniform":
            uniform_weight_style(line)

    def plot_condnum(ax, conds, name):
        """Plot condition numbers as a function of r on the given axes."""
        rs = np.arange(2, len(conds)+2)
        line = ax.semilogy(rs, conds, '-', lw=.5, label=name)[0]
        if name == "Uniform":
            uniform_weight_style(line)

    def plot_ranks(ax, ranks, name):
        """Plot ranks of data matrix as a function of d on the given axes."""
        rs = np.arange(2, len(ranks)+2)
        ds = 1 + rs + rs*(rs + 1)//2 + 1
        line = ax.plot(ds, ranks, '-', lw=.5, label=name)[0]
        if name == "Uniform":
            uniform_weight_style(line)
            ax.plot(ds, ds, '-', lw=.25, color="gray")

    # Figure 1: Normalized weight choices
    fig1, [ax11, ax12] = plt.subplots(1, 2, figsize=(12,4), sharex=True)
    # Figure 2: SVD value decay of bases
    fig2, [ax21, ax22] = plt.subplots(1, 2, figsize=(12,4), sharex=True)
    # Figure 3: condition number of D
    fig3, [ax31, ax32] = plt.subplots(1, 2, figsize=(12,4), sharex=True)
    # Figure 4: results for DQs (SVD value decay, cond(D))
    fig4, [ax41, ax42] = plt.subplots(1, 2, figsize=(12,4))

    with h5py.File(filename, 'r') as hf:

        t = hf["temporal/sigma_01/time"][:]
        # Plot non-uniform weight choices.
        for groupname, name in [("spatial/sigma_04", r"$\sigma = 4$"),
                                ("spatial/sigma_16", r"$\sigma = 16$"),
                                ("spatial/sigma_64", r"$\sigma = 64$"),
                                ("plain/sigma_0", "Euclidean"),
                                ("temporal/sigma_04", r"$\sigma = 4$"),
                                ("temporal/sigma_16", r"$\sigma = 16$"),
                                ("temporal/sigma_64", r"$\sigma = 64$"),
                                ("DQs/order_2", "order 2"),
                                ("DQs/order_4", "order 4"),
                                ("DQs/order_6", "order 6")]:
            if groupname.startswith(("spatial", "plain")):
                axes = [ax11, ax21, ax31]
            elif groupname.startswith("temporal"):
                axes = [ax12, ax22, ax32]
            elif groupname.startswith("DQs"):
                axes = [None, ax41, ax42]
            else:
                raise ValueError(f"invalid groupname '{groupname}'")
            group = hf[groupname]

            if axes[0]:
                plot_weights(axes[0], t, group["weights"][:], name)
            plot_svdvals(axes[1], group["svdvals"][:], name)
            plot_condnum(axes[2], group["dataconds"][:], name)
            # plot_ranks(axes[3], group["dataranks"][:], name)

        # Plot uniform weighting results on each subplot.
        uniform = hf["temporal/sigma_01"]
        name = "Uniform"
        weights = uniform["weights"][:]
        for ax in [ax11, ax12]:
            plot_weights(ax, t, weights, name)

        svdvals = uniform["svdvals"][:]
        for ax in [ax21, ax22, ax41]:
            plot_svdvals(ax, svdvals, name)

        conds = uniform["dataconds"][:]
        for ax in [ax31, ax32, ax42]:
            plot_condnum(ax, conds, name)

        # ranks = uniform["dataranks"][:]
        # plot_ranks(ax41, ranks, name)
        # plot_ranks(ax42, ranks, name)

    # Format axes.
    for ax in [ax11, ax12]:
        # ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Normalized weights")
        ax.legend(loc="upper right", fontsize="large")
    ax11.set_ylim(3e-5, 1.2e-4)
    for ax in [ax21, ax22, ax41]:
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized POD singular values")
        ax.legend(loc="lower left" if ax is ax41 else "upper right",
                  fontsize="large")
        ax.set_ylim(9e-3, 1.05)
    for ax in [ax31, ax32, ax42]:
        ax.set_xlabel(r"Number of POD basis vectors $r$")
        ax.set_ylabel(r"Condition number of data matrix $\mathbf{D}$")
        ax.legend(loc="lower right", fontsize="large")
        ax.set_ylim(1e7, 3e14)
    for ax in [ax11, ax21, ax31]:
        ax.set_title("Gaussian Kernel Weighting")
    for ax in [ax12, ax22, ax32]:
        ax.set_title("Temporal Decay Weighting")

    # Save figures.
    plt.figure(fig1.number)
    utils.save_figure("weights.pdf")
    plt.figure(fig2.number)
    utils.save_figure("weights2decay.pdf")
    plt.figure(fig3.number)
    utils.save_figure("weights2conditioning.pdf")
    plt.figure(fig4.number)
    utils.save_figure("difference_quotients.pdf")


def migrate_data_for_testing(k, groupname, filename="experiment.h5"):
    """Move the data in the specified file to the main data files
    for testing with, e.g., step3_train.py.

    Parameters
    ----------
    k : int
        Number of training snapshots.

    groupname : str
        Name of the group from which to pull data.

    filename : str
        Name of the file from which to pull data.
    """
    filename = os.path.join(config.BASE_FOLDER, config.TRNFMT(k), filename)
    with h5py.File(filename, 'r') as hf:
        group = hf[groupname]

        # weights = group["weights"][:]
        t = group["time"][:]
        scales = group["scales"][:]
        qbar = group["mean_snapshot"][:]
        V = group["basis"][:]
        svdvals = group["svdvals"][:]
        Q_ = group["projected_snapshots"][:]
        Qdot_ = group["projected_derivatives"][:]

    # Save the POD basis.
    save_path = config.basis_path(k)
    with utils.timed_block("Saving POD basis"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("basis", data=V)
            hf.create_dataset("svdvals", data=svdvals)
            hf.create_dataset("mean", data=qbar)
            hf.create_dataset("scales", data=scales)
    logging.info(f"POD bases of rank {V.shape[1]} saved to {save_path}.")

    # Save the projected training data.
    save_path = config.projected_data_path(k)
    with utils.timed_block("Saving projected data"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", data=Q_)
            hf.create_dataset("ddt", data=Qdot_)
            hf.create_dataset("time", data=t)
    logging.info(f"Projected data saved to {save_path}.\n")


# Tests =======================================================================

def test_center_snapshots(n, k):
    """Test center_snapshots()."""
    X = np.random.randint(-10, 10, (n,k)).astype(float)
    weights = .1 + np.random.random(k)
    assert weights.shape == (k,)

    X_centered, xbar = center_snapshots(X, weights)
    assert X_centered is not X
    assert X_centered.shape == X.shape
    assert xbar.shape == (X.shape[0],)
    assert np.allclose(X_centered @ weights, 0)


# =============================================================================
if __name__ == "__main__":
    pass
    # main(15000, 100, [1, 4, 16, 64])
    # plot_results(15000)
    # migrate_data_for_testing(15000, "temporal/sigma_01")
