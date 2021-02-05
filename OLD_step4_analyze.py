# step4_analyze.py
"""Simulate learned ROMs and visualize results.

Figures are saved in a folder given by config.figures_path().

Plot Types
----------
--L-curve: Data misfit vs solution norm with varied regularizations.
    One figure produced per mode specified.

--time-traces: Overlay of data and ROM estimates for each state variable at
    specified monitoring locations (one figure for each of the 8 variables).
    Use --monitor-locations to specify one or more monitoring locations.

--basis-vs-error: Basis size (ROM dimension) vs average errors in each
    variable at specified time indices.
    Use --timeindex to specify one or more snapshot time indices.

--statistical-features: Overlay of data and ROM estimates
    for Temperature integrals (sums)
    of each species concentration.

Examples
--------
# Plot L-curves for ROMs of dimension 24 and 29, learned from 10000 snaphots.
$ python3 step4_analyze.py 10000 --L-curve --modes 24 29

# Plot basis size vs. average error in each variable for ROMs trained with
# regularization factors of 9e4 from 5000 snapshots at time index 2500.
$ python3 step4_analyze.py 5000 --basis-vs-error --modes 17 30 --moderange
                                --regularization 9e4 --timeindex 2500

# Plot time traces in each variable at default monitoring locations for
# the ROM of dimension 24, trained with a regularization factor of 9e4
# from 10000 snapshots.
$ python3 step4_analyze.py 10000 --time-traces --modes 24 -reg 9e4

# Plot species integrals for ROMs of dimension 24 and 29,
# trained with regularization factors of 6e4 and 1e5 from 15000 snapshots.
$ python3 step4_analyze.py 15000 --species-integral -r 24 29 -reg 6e4 1e5
"""
import os
import h5py
import logging
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc


def L_curve(trainsize, r):
    """Plot the L-curve (solution norm vs residual norm) for a collection
    of ROMs of the same dimension.
    Results are saved as Lcurve_k<trainsize>_r<r>.pdf.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROMs. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r : int
        The dimension of the ROMs. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.
    """
    # Find and load all of the stored ROMs.
    regs, roms = utils.load_all_roms_r(trainsize, r)

    # Get operator norms, and residuals of stored ROMs.
    residuals = [rom.misfit_ for rom in roms]
    norms = [rom.operator_norm_ for rom in roms]

    # Plot the results.
    fig, ax = plt.subplots(1, 1, figsize=(10,4))

    # # Discrete points
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(regs)))
    # for norm, resid, reg, c in zip(norms, residuals, regs, colors):
    #     ax.loglog([resid], [norm], '.', color=c, ms=20,
    #               label=fr"$\lambda = {reg:e}$")
    # ax.legend(loc="lower left", ncol=2)

    # Continuous colored line
    x, y, z = np.array(residuals), np.array(norms), np.array(regs)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='Spectral_r',
                        norm=LogNorm(z.min(),z.max()))
    lc.set_array(z)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax, label=r"regularization $\lambda$")
    # cbar.ax.set_ylabel(r"regularization $\lambda$", rotation=270)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$||\mathbf{D}\mathbf{O}^\top - \dot{\hat{\mathbf{X}}}||_F^2$")
    ax.set_ylabel(r"$||\mathbf{O}^\top||_F^2$")
    ax.grid()
    ax.set_title(f"$r = {r:d}$")

    utils.save_figure(f"Lcurve"
                      f"_{config.TRNFMT(trainsize)}_{config.DIMFMT(r)}.pdf")


def simulate_rom(trainsize, r, reg):
    """Load everything needed to simulate a given ROM, simulate the ROM,
    and return the simulation results and other useful things. Raise an
    Exception if any of the ingredients are missing.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM.

    reg : float
        Regularization factors used to train the ROM.

    Returns
    -------
    t : (nt,) ndarray
        Time domain

    V : (config*NUM_ROMVARS*config.DOF,r) ndarray
        Basis

    scales : (NUM_ROMVARS,4) ndarray
        Information for how the data was scaled

    x_rom : (nt,r) ndarray
        Prediction results from the ROM.
    """
    # Load the time domain, basis, initial conditions, and trained ROM.
    t = utils.load_time_domain()
    V, scales = utils.load_basis(trainsize, r)
    X_, _, _ = utils.load_projected_data(trainsize, r)
    rom = utils.load_rom(trainsize, r, reg)

    # Simulate the ROM over the full time domain.
    with utils.timed_block(f"Simulating ROM with r={r:d}, "
                           f"reg={reg:e} over full time domain"):
        x_rom = rom.predict(X_[:,0], t, config.U, method="RK45")

    return t, V, scales, x_rom


def basis_size_vs_error(trainsize, num_modes, regs, timeindices):
    """For each pair of number of modes and regularization, simulate the
    corresponding ROM trained with `trainsize` snapshots and compare the
    results of the `timeindex`th step to the original data.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROMs. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    num_modes : int or list(int)
        The dimension of the ROMs. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    regs : float or list(float)
        The regularization factors used to train each ROM. This sequence must
        have the same length as num_modes.

    timeindices : list(int) or ndarray(int)
        Indices in the time domain at which to compute the errors.
    """
    # Parse arguments
    if np.isscalar(num_modes):
        num_modes = [num_modes]
    if np.isscalar(regs):
        regs = [regs]*len(rs)
    if len(num_modes) != len(regs):
        raise ValueError("same number of modes and regularizations required")
    timeindices = np.array(timeindices)

    # Load and unpack the true results.
    t = utils.load_time_domain()
    data, _ = utils.load_gems_data(cols=timeindices)
    with utils.timed_block("Lifting true snapshots for comparison"):
        true_snaps = dproc.lift(data)
    V, scales = utils.load_basis(trainsize, max(num_modes))

    # Initialize the figures (one for each time index).
    figs_axes = [plt.subplots(3, 3, figsize=(4,6), sharex=True)
                 for _ in range(timeindices.size)]

    # Get results for each (r,reg) pairing.
    simtime = timeindices.max()
    tsim = t[:simtime+1]
    for r, reg in zip(num_modes, regs):
        # Load the initial conditions and scales.
        X_, _, _ = utils.load_projected_data(trainsize, r)

        # Load the appropriate ROM.
        try:
            rom = utils.load_rom(trainsize, r, reg)
        except utils.DataNotFoundError as e:
            print(f"ROM file not found: {e} (skipping)")
            continue

        # Simulate the ROM over the time domain.
        with utils.timed_block(f"Simulating ROM with r={r:d}, reg={reg:e}"):
            x_rom = rom.predict(X_[:,0], tsim, config.U, method="RK45")
            if np.any(np.isnan(x_rom)) or x_rom.shape[1] < simtime:
                print(f"ROM with r = {r:d} and reg = {reg:e} unstable!")
                continue

        # Reconstruct the results (all variables, one snapshot).
        with utils.timed_block("Reconstructing simulation results"):
            pred_snaps = dproc.unscale(V[:,:r] @ x_rom[:,timeindices], scales)

        # Compute and plot the error in each variable.
        with utils.timed_block("Computing / plotting errors in each variable"):
            for j,(fig,axes) in enumerate(figs_axes):
                true_snap = true_snaps[:,j]
                pred_snap = pred_snaps[:,j]
                for k,varname in enumerate(config.ROM_VARIABLES):
                    truevar = dproc.getvar(varname, true_snap)
                    predvar = dproc.getvar(varname, pred_snap)
                    if varname in ["p", "T"]:
                        # Pressure and temperature: relative error.
                        err = utils.mean_relative_error(truevar, predvar)
                    else:
                        # All others: normalized absolute error.
                        err = utils.mean_normalized_absolute_error(truevar,
                                                                   predvar)
                    axes.flatten()[k].plot([r], [err], marker='.', ms=2,
                                           color=config.ROM_STYLE["color"])

    for j,(fig,axes) in enumerate(figs_axes):
        for ax, var in zip(axes.flatten(), config.ROM_VARIABLES):
            # Do linear regression for each data set.
            # _x = np.array([line.get_xdata()[0] for line in ax.lines])
            # _y = np.array([line.get_ydata()[0] for line in ax.lines])
            # _m, _b = stats.linregress(_x, _y)[:2]
            # ax.plot(_x, _m*_x + _b, 'k-', lw=.5)

            ax.set_ylabel(f"{config.VARTITLES[var]} Error", fontsize=14)
            ax.grid(lw=.5, alpha=.25)               # Light grid overlay.
            ax.locator_params(axis='y', nbins=3)    # 3 yticks on each subplot.

        for ax in axes[1,:]:
            ax.set_xlabel("Basis Size", fontsize=14)

        # Set figure title and save.
        idx = timeindices[j]
        # fig.suptitle(f"Average State Errors at $t = {t[idx]:f}$", fontsize=18)
        plt.figure(fig.number)
        plt.tight_layout()
        utils.save_figure(f"basisVerror"
                          f"_{config.TRNFMT(trainsize)}_snap{idx:05d}.pdf")


def time_traces(trainsize, r, reg, elems):
    """Plot the time trace of each variable in the original data at the monitor
    location, and the time trace of each variable of the ROM reconstruction at
    the same locations. One figure is generated per variable.

    Parameters
    ----------
    trainsize : int
        The number of snapshots that were used to train the ROM.

    r : int
        The dimension of the ROMs This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    regs : float
        The regularization factors used to train the ROM.

    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute the time traces.
    """
    # Get the indicies for each variable.
    elems = np.atleast_1d(elems)
    nelems = elems.size
    nrows = (nelems // 2) + (1 if nelems % 2 != 0 else 0)
    elems = np.concatenate([elems + i*config.DOF
                            for i in range(config.NUM_ROMVARS)])

    # Load and lift the true results.
    data, _ = utils.load_gems_data(rows=elems[:nelems*config.NUM_GEMSVARS])
    with utils.timed_block("Lifting GEMS time trace data"):
        traces_gems = dproc.lift(data)

    # Load and simulate the ROM.
    t, V, scales, x_rom = simulate_rom(trainsize, r, reg)

    # Reconstruct and rescale the simulation results.
    simend = x_rom.shape[1]
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom = dproc.unscale(V[elems] @ x_rom, scales)

    # Save a figure for each variable.
    xticks = np.arange(t[0], t[-1]+.001, .002)
    for i,var in enumerate(config.ROM_VARIABLES):
        fig, axes = plt.subplots(nrows, 2 if nelems > 1 else 1,
                                 figsize=(9,3*nrows), sharex=True)
        axes = np.atleast_2d(axes)
        for j, ax in zip(range(nelems), axes.flat):
            idx = j + i*nelems
            ax.plot(t, traces_gems[idx,:], lw=1, **config.GEMS_STYLE)
            ax.plot(t[:simend], traces_rom[idx,:], lw=1, **config.ROM_STYLE)
            ax.axvline(t[trainsize], color='k', lw=1)
            ax.set_xlim(t[0], t[-1])
            ax.set_xticks(xticks)
            ax.set_title(f"Location ${j+1}$", fontsize=12)
            ax.locator_params(axis='y', nbins=2)
        for ax in axes[-1,:]:
            ax.set_xlabel("Time [s]", fontsize=12)
        for ax in axes[:,0]:
            ax.set_ylabel(config.VARLABELS[var], fontsize=12)
        # Specific axis limits (TODO: remove from public version)
        if var == "p":
            for ax in axes.flat:
                ax.set_ylim(8.5e5, 1.35e6)
        elif var == "T":
            axes[0,0].set_ylim(1.45e3, 2.45e3)
            axes[0,1].set_ylim(4.5e2, 3.5e3)
            axes[1,0].set_ylim(0, 2.75e3)
            axes[1,1].set_ylim(6.6e2, 7.6e2)
        # Legend on the right.
        fig.tight_layout(rect=[0, 0, .85, 1])
        leg = axes[0,0].legend(loc="center right", fontsize=14,
                               bbox_to_anchor=(1,.5),
                               bbox_transform=fig.transFigure)
        for line in leg.get_lines():
            line.set_linewidth(2)
        utils.save_figure("timetrace"
                          f"_{config.TRNFMT(trainsize)}"
                          f"_{config.DIMFMT(r)}"
                          f"_{config.REGFMT(reg)}_{var}.pdf")


def save_statistical_features():
    """Compute the (spatial) mean temperatures on the full time domain and
    save them for later. This only needs to be done once."""
    # Load the full data set.
    gems_data, t = utils.load_gems_data()

    # Lift the data (convert to molar concentrations).
    with utils.timed_block("Lifting GEMS data"):
        lifted_data = dproc.lift(gems_data)

    # Compute statistical features.
    with utils.timed_block("Computing statistical features of variables"):
        mins, maxs, sums, stds, means = {}, {}, {}, {}, {}
        for var in config.ROM_VARIABLES:
            val = dproc.getvar(var, lifted_data)
            mins[var] = val.min(axis=0)
            maxs[var] = val.max(axis=0)
            sums[var] = val.sum(axis=0)
            stds[var] = val.std(axis=0)
            means[var] = sums[var] / val.shape[0]

    # Save the data.
    data_path = config.statistical_features_path()
    with utils.timed_block("Saving statistical features"):
        with h5py.File(data_path, 'w') as hf:
            for var in config.ROM_VARIABLES:
                hf.create_dataset(f"{var}_min", data=mins[var])
                hf.create_dataset(f"{var}_max", data=maxs[var])
                hf.create_dataset(f"{var}_sum", data=sums[var])
                hf.create_dataset(f"{var}_std", data=stds[var])
                hf.create_dataset(f"{var}_mean", data=means[var])
            hf.create_dataset("time", data=t)
    logging.info(f"Statistical features saved to {data_path}")


def statistical_features(trainsize, r, reg):
    """Plot spatially averaged temperature and spacially itegrated (summed)
    species concentrations over the full time domain.

    Parameters
    ----------
    trainsize : int
        The number of snapshots that were used to train the ROM.

    r : int
        The dimension of the ROM. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    reg : float
        The regularization factor used to train each ROM.
    """
    # Load the true results.
    keys = [f"{var}_mean" for var in config.ROM_VARIABLES[:4]]
    keys += ["T_mean"] + [f"{var}_sum" for var in config.SPECIES]
    feature_gems, t = utils.load_spatial_statistics(keys)
    keys = np.reshape(keys, (4,2))

    # Load and simulate the ROM.
    t, V, scales, x_rom = simulate_rom(trainsize, r, reg)
    
    # Initialize the figure.
    fig, axes = plt.subplots(keys.shape[0], keys.shape[1],
                             figsize=(9,6), sharex=True)
    
    # Calculate and plot the results.
    for ax,key in zip(axes.flat, keys.flat):
        ax.plot(t, feature_gems[key], **config.GEMS_STYLE)
        var, action = key.split('_')
        with utils.timed_block(f"Reconstructing {action}({var})"):
            x_rec = dproc.unscale(dproc.getvar(var, V) @ x_rom, scales, var)
            feature_rom = eval(f"x_rec.{action}(axis=0)")
        ax.plot(t[:x_rom.shape[1]], feature_rom, **config.ROM_STYLE)
        ax.axvline(t[trainsize], lw=1, color='k')
        ax.set_ylabel(config.VARLABELS[var])
        ax.locator_params(axis='y', nbins=2)

    # Set titles, labels, ticks, and draw a single legend.
    for ax in axes[-1,:]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=12)
    ax[0,0].set_title("Spatial Averages", fontsize=14)
    ax[0,1].set_title("Spatial Integrals", fontsize=14)

    # Legend on the right.
    fig.tight_layout(rect=[0, 0, .85, 1])
    leg = axes[0,0].legend(loc="center right", fontsize=14,
                           bbox_to_anchor=(1,.5),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)

    utils.save_figure(f"statfeatures
                      f"_{config.TRNFMT(trainsize)}"
                      f"_{config.DIMFMT(r)}"
                      f"_{config.REGFMT(reg)}.pdf")

# =============================================================================

def main(trainsize, num_modes, regs, elems, timeindices,
         plotLcurve=False, plotBasisSize=False,
         plotTimeTrace=False, plotStatisticalFeatures=False):
    """Make the indicated visualization.

    Parameters
    ----------
    trainsize : int
        The number of snapshots that were used to train the ROM.

    num_modes : int or list(int)
        The dimension of the ROMs. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    regs : float or list(float)
        The regularization factors used to train each ROM.

    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute time traces.

    timeindices : list(int) or ndarray(int)
        Indices in the time domain at which to compute basis errors.
    """
    utils.reset_logger(trainsize)

    # L-curves (ROMs of same dimension, different regularization).
    if plotLcurve:
        logging.info("L CURVE")
        L_curve(trainsize, num_modes)
        return

    # Basis size vs Error (ROMs of different dimension, same regularization).
    if plotBasisSize:
        logging.info("BASIS SIZE VS ERROR")
        basis_size_vs_error(trainsize, num_modes, regs, timeindices)
        return

    # Time traces (single ROM, several monitoring locations).
    if plotTimeTrace:
        logging.info("PLOTTING TIME TRACES")
        time_traces(trainsize, num_modes, regs, elems)
        return


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE --L-curve
                                  --modes R
        python3 {__file__} TRAINSIZE --basis-vs-error
                                  --modes R [...] [--moderange]
                                  --regularization REG [...] [--regrange NREGS]
                                  --timeindex T [...]
        python3 {__file__} TRAINSIZE --time-traces
                                  --modes R --regularization REG
                                  --monitor-location M [...]
        python3 {__file__} TRAINSIZE --statistical-features
                                  --modes R --regularization REG"""
    # Positional arguments
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")

    # Mutually exclusive plotting options (only one allowed at a time).
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--L-curve", action="store_true",
                       help="plot L-curves for the given basis sizes")
    group.add_argument("--time-traces", action="store_true",
                       help="plot time traces for the given "
                             "basis sizes and regularization factors "
                             "at the specified monitoring locations")
    group.add_argument("--basis-vs-error", action="store_true",
                       help="plot basis size vs average error for the given "
                             "regularization factors ")
    group.add_argument("--statistical-features", action="store_true",
                       help="plot spatial averages and species integrals "
                            "for the ROM with the given basis size and "
                            "regularization factors")

    # Other keyword arguments
    parser.add_argument("-r", "--modes", type=int, nargs='+',
                        default=[],
                        help="number of POD modes used to project data")
    parser.add_argument("--moderange", action="store_true",
                        help="if two modes given, treat them as min, max "
                             "and set modes = integers in [min, max].")
    parser.add_argument("-reg", "--regularization", type=float, nargs='+',
                        default=[],
                        help="regularization factor used in ROM training")
    parser.add_argument("-snap", "--timeindex", type=int, nargs='*',
                        default=list(range(0,34100,100)),
                        help="snapshot index for basis vs error plots")
    parser.add_argument("-elm", "--monitor-location", type=int, nargs='+',
                        default=config.MONITOR_LOCATIONS,
                        help="monitor locations for time trace plots")

    # Do the main routine.
    args = parser.parse_args()
    if args.moderange and len(args.modes) == 2:
        args.modes = list(range(args.modes[0], args.modes[1]+1))

    main(trainsize=args.trainsize,
         num_modes=args.modes,
         regs=args.regularization,
         elems=args.monitor_location,
         timeindices=args.timeindex,
         plotLcurve=args.L_curve,
         plotBasisSize=args.basis_vs_error,
         plotTimeTrace=args.time_traces,
         plotStatisticalFeatures=args.statistical_features)
