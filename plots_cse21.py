from plots import *
import plots
    
    
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


def averages(trainsize, num_modes, regs,
             cutoff=60000, filename="spatial_averages.pdf"):

    # Load GEMS statistical features.
    keys = ["CH4_sum", "CO2_sum"]
    features_gems, t = utils.load_spatial_statistics(keys, cutoff)

    # Initialize the figure.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    axes = np.atleast_2d(axes)

    # Load basis, training data, ROM, then simulate and reconstruct.
    t, V, scales, q_rom = step4.simulate_rom(trainsize, num_modes, regs,cutoff)
    with utils.timed_block(f"Processing k={trainsize:d}, r={num_modes:d}"):
        for ax,key in zip(axes.flat, keys):
            features_pred = step4.get_feature(key, q_rom, V, scales)
            ax.plot(t, features_gems[key], **config.GEMS_STYLE)
            ax.plot(t, features_pred, **config.ROM_STYLE)
            ax.axvline(t[trainsize], lw=2, color='k')

    # Format the figure.
    sep = '\n'
    ylabels = []
    for key in keys:
        v,action = key.split('_')
        if action == "sum":
            ylabels.append(f"{config.VARTITLES[v]} Concentration{sep}"
                           f"Integral [{config.VARUNITS[v]}]")
        elif action == "mean":
            ylabels.append(f"Spatially Averaged{sep}{config.VARLABELS[v]}")

    plots._format_subplots(fig, axes, [num_modes]*2, ylabels, numbers=False,
                           ps=filename.endswith(".ps"))

    # Set custom y limits for plots in the publication.
    for key, ax in zip(keys, axes.flatten()):
        if key == "T_mean":
            ax.set_ylim(8.25e2, 1.2e3)
        if key == "CH4_sum":
            ax.set_ylim(1e3, 1.6e3)
        if key == "CO2_sum":
            ax.set_ylim(35, 105)
        if key == "O2_sum":
            ax.set_ylim(1.15e3, 1.6e3)
            ax.set_yticks([1.2e3, 1.5e3])

    utils.save_figure(filename)



# -----------------------------------------------------------------------------

def main():
    custom_traces(trainsizes=[      20000,         20000,           20000],
                   num_modes=[         43,            43,              43],
                        regs=[(5000,5000), (316, 18199), (200000,200000)],
                  variables=["p"],
                  locs=[config.MONITOR_LOCATIONS[1]],
                  loclabels=[2],
                  cutoff=60000,
                  filename="multireg.pdf")

    traces(20000, 43, (316,18199))
    averages(20000, 43, (316,18199))

    projection_errors(trainsize=20000, r=43, regs=(316,18199),
                      variables=["p", "T"], cutoff=60000,
                      filename="projerrors.pdf")


# =============================================================================
if __name__ == "__main__":
    # main()
    averages(20000, 43, (316,18199))
