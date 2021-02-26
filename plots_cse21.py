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


def averages(trainsize, r, regs, cutoff=60000, filename="spatial_averages.pdf"):
    """Plot a comparison between OpInf and POD-DEIM.
    | Feature 1 | | Feature 2 |
    """
    # Load the relevant GEMS data.
    keys = ["O2_sum", "CO2_sum"]
    assert len(keys) == 2                   # two spatial features.
    specs_gems, t2 = utils.load_spatial_statistics(keys, k=cutoff)

    # Get OpInf simulation results and extract relevant data.
    t_rom, V, scales, q_rom = step4.simulate_rom(trainsize, r, regs, cutoff)
    with utils.timed_block("Extracting OpInf features"):
        specs_rom = {k: step4.get_feature(k, q_rom, V, scales) for k in keys}

    # Load POD-DEIM data and extract relevant data.
    data_deim, t_deim = poddeim.load_data(cols=cutoff)
    with utils.timed_block("Extracting POD-DEIM features"):
        lifted_deim = dproc.lift(data_deim)
        del data_deim
        specs_deim = {k: step4.get_feature(k, lifted_deim) for k in keys}

    # Initialize the figure.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

    # Plot results.
    with utils.timed_block("Plotting results"):

        # Spatial features: bottom plots.
        for key, ax in zip(keys, axes.flat):
            ax.plot(t2, specs_gems[key], **config.GEMS_STYLE)
            ax.plot(t_deim, specs_deim[key], **poddeim.STYLE)
            ax.plot(t_rom, specs_rom[key], **config.ROM_STYLE)
            ax.axvline(t2[trainsize], lw=2, color='k')
            plots._format_y_axis(ax)
            plots._format_x_axis(ax)

    fig.tight_layout(rect=[0, .15, 1, 1])
    fig.subplots_adjust(hspace=.6, wspace=.25)

    # Create the legend, centered at the bottom of the plot.
    labels = ["GEMS", "POD-DEIM", "OpInf"]
    plots._format_legend(fig, axes[0], labels)

    # Set titles and labels.
    for key, ax in zip(keys, axes.flat):
        v,action = key.split('_')
        if action == "sum":
            ax.set_ylabel(f"{config.VARTITLES[v]} Concentration\n"
                           f"Integral [{config.VARUNITS[v]}]",
                           fontsize=MEDFONT, labelpad=2)
        elif action == "mean":
            ax.set_ylabel(f"Spatially Averaged\n{config.VARLABELS[v]}",
                           fontsize=MEDFONT, labelpad=2)
        if key == "T_mean":
            ax.set_ylim(8.25e2, 1.2e3)
        if key == "CH4_sum":
            ax.set_ylim(1e3, 1.6e3)
        if key == "CO2_sum":
            ax.set_ylim(50, 120)
            ax.set_yticks([75, 100])
        if key == "O2_sum":
            ax.set_ylim(1.2e3, 1.6e3)
            ax.set_yticks([1.3e3, 1.5e3])

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
    compare_poddeim(20000, 43, (316,18199))

    projection_errors(trainsize=20000, r=43, regs=(316,18199),
                      variables=["p", "T"], cutoff=60000,
                      filename="projerrors.pdf")


# =============================================================================
if __name__ == "__main__":
    main()

