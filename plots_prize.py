from plots import *
import plots


def custom_traces(trainsizes, num_modes, regs,
                  variables=(), locs=(), loclabels=None, keys=(),
                  cutoff=None, filename="trace.pdf"):
    """Draw a grid of subplots where each row corresponds to a different
    ROM (trainsizes[i], num_modes[i], regs[i]) and each column is for a
    specific time trace (variables, locs) or statistical features (keys).
    """
    # Load GEMS time trace data for the given indices and variables.
    if variables:
        assert len(variables) == len(locs)
        locs = np.array(locs)
        nlocs = locs.size
        locs = np.concatenate([locs + i*config.DOF
                               for i in range(config.NUM_ROMVARS)])
        if loclabels is None:
            loclabels = [i+1 for i in range(nlocs)]

        # Load and lift the GEMS time trace results.
        gems_locs = locs[:nlocs*config.NUM_GEMSVARS]
        data, t = utils.load_gems_data(rows=gems_locs, cols=cutoff)
        with utils.timed_block("Lifting GEMS time trace data"):
            traces_gems = dproc.lift(data)
    else:
        nlocs = 0

    # Load GEMS statistical features.
    if keys:
        if isinstance(keys, str):
            keys = [keys]
        features_gems, t = utils.load_spatial_statistics(keys, cutoff)
        if len(keys) == 1:
            features_gems = {keys[0]: features_gems}

    # Initialize the figure.
    nrows, ncols = len(trainsizes), nlocs + len(keys)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(9,7))
    if nrows == 1:
        axes = np.atleast_2d(axes)
    elif ncols == 1:
        axes = axes.reshape((-1,1))

    for i, (trainsize,r,reg) in enumerate(zip(trainsizes, num_modes, regs)):

        # Load basis, training data, ROM, then simulate and reconstruct.
        t, V, scales, q_rom = step4.simulate_rom(trainsize, r, reg, cutoff)
        with utils.timed_block(f"Processing k={trainsize:d}, r={r:d}"):

            if variables:   # First len(locs) columns: variable traces.
                traces_pred = step4.get_traces(locs, q_rom, V, scales)
                for j,var in enumerate(variables):
                    romvar = dproc.getvar(var, traces_pred)
                    gemvar = dproc.getvar(var, traces_gems)
                    axes[i,j].plot(t, gemvar[j,:], **config.GEMS_STYLE)
                    axes[i,j].plot(t[:romvar.shape[1]], romvar[j,:],
                                   **config.ROM_STYLE)
                    axes[i,j].axvline(t[trainsize], lw=2, color='k')

            if keys:        # Last len(keys) columns: statistical features.
                for ax,key in zip(axes[i,nlocs:], keys):
                    features_pred = step4.get_feature(key, q_rom, V, scales)
                    ax.plot(t, features_gems[key], **config.GEMS_STYLE)
                    ax.plot(t, features_pred, **config.ROM_STYLE)
                    ax.axvline(t[trainsize], lw=2, color='k')

    # Format the figure.
    if variables:
        for ax, num in zip(axes[0,:nlocs], loclabels):
            ax.set_title(fr"Location ${num}$", fontsize=MEDFONT)
        ylabels = [config.VARLABELS[var] for var in variables]
    else:
        ylabels = []
    if keys:
        sep = ' ' if nrows > 2 else '\n'
        for key in keys:
            v,action = key.split('_')
            if action == "sum":
                ylabels.append(f"{config.VARTITLES[v]} Concentration{sep}"
                               f"Integral [{config.VARUNITS[v]}]")
            elif action == "mean":
                ylabels.append(f"Spatially Averaged{sep}{config.VARLABELS[v]}")

    plots._format_subplots(fig, axes, num_modes, ylabels, numbers=False,
                           ps=filename.endswith(".ps"))
    fig.subplots_adjust(hspace=.2)

    # Set custom y limits for plots in the publication.
    # TODO: remove in public version.
    if len(variables) > 0 and variables[0] == "p" and loclabels[0] == 1:
        for ax in axes[:,0]:
            ax.set_ylim(9.5e5, 1.35e6)
    if len(variables) > 0 and variables[0] == "p" and loclabels[0] == 2:
        for ax in axes[:,0]:
            ax.set_ylim(1e6, 1.3e6)
    if len(variables) > 0 and variables[0] == "CH4":
        for ax in axes[:,0]:
            ax.set_ylim(4e-3, 1.9e-2)
            ax.set_yticks([1e02, 1.5e-2])
    if len(variables) > 0 and variables[0] == "O2":
        for ax in axes[:,0]:
            ax.set_ylim(-2e-3, 1.2e-2)
    if len(variables) > 1 and variables[1] == "T":
        for ax in axes[:,1]:
            ax.set_ylim(5e2, 3.5e3)
            ax.set_yticks([1e3, 3e3])
    if len(variables) > 1 and variables[1] == "vx" and loclabels[1] == 2:
        for ax in axes[:,1]:
            ax.set_ylim(-240, 190)
    if len(variables) > 1 and variables[1] == "vx" and loclabels[1] == 3:
        for ax in axes[:,1]:
            ax.set_ylim(-5, 10)
    if len(variables) > 1 and variables[1] == "vx" and loclabels[1] == 4:
        for ax in axes[:,1]:
            ax.set_ylim(-50, 180)
    if len(variables) > 2 and variables[2] == "vx":
        for ax in axes[:,2]:
            ax.set_ylim(-5, 10)
    if len(variables) > 3 and variables[3] == "CH4":
        for ax in axes[:,3]:
            ax.set_ylim(-1e-24, 5e-24)
    if len(variables) > 3 and variables[3] == "O2":
        for ax in axes[:,3]:
            ax.set_ylim(.05, .065)

    if len(keys) > 0 and keys[0] == "T_mean":
        for ax in axes[:,nlocs+0]:
            ax.set_ylim(8.25e2, 1.2e3)
    for i in range(len(keys)):
        if keys[i] == "CH4_sum":
            for ax in axes[:,nlocs+i]:
                ax.set_ylim(1.1e3, 1.5e3)
        if keys[i] == "CO2_sum":
            for ax in axes[:,nlocs+i]:
                ax.set_ylim(45, 95)
        if keys[i] == "O2_sum":
            for ax in axes[:,nlocs+i]:
                ax.set_ylim(1.15e3, 1.6e3)
                ax.set_yticks([1.2e3, 1.5e3])

    utils.save_figure(filename)

    
    
def traces(trainsize, num_modes, regs,
           cutoff=60000, filename="traces.pdf"):

    # Load the desired GEMS time trace data.
    variables = ["p", "vx", "p", "vx"]
    locations = [1, 3, 2, 4]
    
    locs = []
    for i,var in enumerate(variables):
        offset = config.GEMS_VARIABLES.index(var)*config.DOF
        locs.append(config.MONITOR_LOCATIONS[locations[i]-1] + offset)
    locs = np.array(locs)
    nlocs = locs.size

    # Load the GEMS time trace results.
    traces_gems, t = utils.load_gems_data(rows=locs, cols=cutoff)

    # Initialize the figure.
    fig, axes = plt.subplots(2, 2, figsize=(18, 6), sharex=True)
    # fig, axes = plt.subplots(4, 1, figsize=(9, 15), sharex=True)
    # axes = axes.reshape((4,1))

    # Load basis, training data, ROM, then simulate and reconstruct.
    t, V, scales, q_rom = step4.simulate_rom(trainsize, num_modes, regs, cutoff)
    traces_pred = V[locs] @ q_rom
    for i,var in enumerate(variables):
        dproc.unscale(traces_pred[i,:], scales, var)
    
    for i,ax in enumerate(axes.flat):
        ax.plot(t, traces_gems[i,:], **config.GEMS_STYLE)
        ax.plot(t, traces_pred[i,:], **config.ROM_STYLE)
        ax.axvline(t[trainsize], lw=2, color='k')
        ax.set_title(fr"Location ${locations[i]}$", fontsize=MEDFONT)

    # Format and save the figure.
    ylabels = [config.VARLABELS["p"], config.VARLABELS["vx"]]
    plots._format_subplots(fig, axes, num_modes, ylabels, numbers=False,
                           ps=filename.endswith(".ps"))
    fig.subplots_adjust(hspace=.35, wspace=.2)
    utils.save_figure(filename)


def main():
    custom_traces(trainsizes=[      20000,         20000,           20000],
                   num_modes=[         43,            43,              43],
                        regs=[(5000,5000), (22841,22841), (200000,200000)],
                  variables=["p"],
                  locs=[config.MONITOR_LOCATIONS[1]],
                  loclabels=[2],
                  cutoff=60000,
                  filename="multireg.pdf")

    traces(20000, 43, (292,18347))

    projection_errors(trainsize=20000, r=43, regs=(292,18347),
                      variables=["p", "T"], cutoff=60000,
                      filename="projerrors.pdf")


if __name__ == "__main__":
    main()
