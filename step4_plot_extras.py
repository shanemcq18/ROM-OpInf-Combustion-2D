# plot_extras.py
"""Functions that used to live in step4_plot.py."""

from step4_plot import *


def veccorrcoef(X, Y):
    """Calculate the (vectorized) linear correlation coefficent,

                     sum_i[(X_i - Xbar)Y_i - Ybar)]
        r = -------------------------------------------------.
            sqrt(sum_i[(X_i - Xbar)^2] sum_i[(Y_i - Ybar)^2])

    This function is equivalent to (but much faster than)
    >>> r = [np.corrcoef(X[:,j], Y[:,j])[0,1] for j in range(X.shape[1])].

    Parameters
    ----------
    X : (n,k) ndarray
        First array of data, e.g., ROM reconstructions of one variable.

    Y : (n,k) ndarray
        Second array of data, e.g., GEMS data of one variable.

    Returns
    -------
    r : (k,) ndarray
        Linear correlation coefficient of X[:,j], Y[:,j] for j = 0, ..., k-1.
    """
    dX = X - np.mean(X, axis=0)
    dY = Y - np.mean(Y, axis=0)
    numer = np.sum(dX*dY, axis=0)
    denom2 = np.sum(dX**2, axis=0) * np.sum(dY**2, axis=0)
    return numer / np.sqrt(denom2)


# Plot routines ===============================================================

def corrcoef(trainsize, r, regs, cutoff=60000):
    """Plot correlation coefficients in time between GEMS and ROM solutions.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM.

    regs : float
        Regularization hyperparameters used to train the ROM.

    cutoff : int
        Numer of time steps to plot.
    """
    # Load and simulate the ROM.
    t, V, scales, q_rom = simulate_rom(trainsize, r, regs, cutoff)

    # Load and lift the true results.
    data, _ = utils.load_gems_data(cols=cutoff)
    with utils.timed_block("Lifting GEMS data"):
        data_gems = dproc.lift(data[:,:cutoff])

    # Initialize the figure.
    fig, axes = plt.subplots(3, 3, figsize=(12,6), sharex=True, sharey=True)

    # Compute and plot errors in each variable.
    for var, ax in zip(config.ROM_VARIABLES, axes.flat):

        with utils.timed_block(f"Reconstructing results for {var}"):
            Vvar = dproc.getvar(var, V)
            gems_var = dproc.getvar(var, data_gems)
            pred_var = dproc.unscale(Vvar @ q_rom, scales, var)

        with utils.timed_block(f"Calculating correlation in {var}"):
            corr = veccorrcoef(gems_var, pred_var)

        # Plot results.
        ax.plot(t, corr, '-', lw=1, color='C2')
        ax.axvline(t[trainsize], color='k')
        ax.axhline(.8, ls='--', lw=1, color='k', alpha=.25)
        ax.set_ylim(0, 1)
        ax.set_ylabel(config.VARTITLES[var])

    # Format the figure.
    for ax in axes[-1,:]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=14)
    fig.suptitle("Linear Correlation Coefficient", fontsize=16)

    # Save the figure.
    utils.save_figure(f"corrcoef"
                      f"_{config.TRNFMT(trainsize)}"
                      f"_{config.DIMFMT(r)}"
                      f"_{config.REGFMT(regs)}.pdf")


def projection_errors(trainsize, rs):
    """Plot spatially averaged projection errors in time.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    rs : list(int)
        Basis sizes to test
    """
    # Load and lift the true results.
    data, t = utils.load_gems_data()
    with utils.timed_block("Lifting GEMS data"):
        data_gems = dproc.lift(data)
    del data

    # Initialize the figure.
    fig, axes = plt.subplots(3, 3, figsize=(12,6), sharex=True)

    # Get projection errors for each r.
    for r in rs:

        # Load the POD basis of rank r.
        V, scales = utils.load_basis(trainsize, r)

        # Shift the data (unscaling done later by chunk).
        if r == rs[0]:
            with utils.timed_block(f"Shifting GEMS data"):
                data_shifted, _ = dproc.scale(data_gems.copy(), scales)

        # Project the shifted data.
        with utils.timed_block(f"Projecting GEMS data to rank-{r} subspace"):
            data_proj = V.T @ data_shifted

        # Compute and plot errors in each variable.
        for var, ax in zip(config.ROM_VARIABLES, axes.flat):

            with utils.timed_block(f"Reconstructing results for {var}"):
                Vvar = dproc.getvar(var, V)
                gems_var = dproc.getvar(var, data_gems)
                proj_var = dproc.unscale(Vvar @ data_proj, scales, var)

            with utils.timed_block(f"Calculating error in {var}"):
                denom = np.abs(gems_var).max(axis=0)
                proj_error = np.mean(np.abs(proj_var-gems_var), axis=0) / denom

            # Plot results.
            ax.plot(t, proj_error, '-', lw=1, label=fr"$r = {r}$")

    # Format the figure.
    for ax in axes[-1,:]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=12)
    for var, ax in zip(config.ROM_VARIABLES, axes.flat):
        ax.axvline(t[trainsize], color='k')
        ax.set_ylabel(config.VARTITLES[var])

    # Make legend centered below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    leg = axes[0,0].legend(ncol=3, fontsize=14, loc="lower center",
                           bbox_to_anchor=(.5, 0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linestyle('-')
        line.set_linewidth(5)

    # Save the figure.
    utils.save_figure(f"projerrors_{config.TRNFMT(trainsize)}.pdf")
