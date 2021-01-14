# bayes.py
"""Bayesian interpretation of Operator Inference for this problem.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc


# Operator Inference posterior samplers =======================================

class OpInfPosterior:
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.
    """
    def __init__(self, means, Sigmas, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------
        means : list of r (d,) ndarrays or (r,d) ndarray
            Mean values for each of the operator entries, i.e., Mean(O).

        Sigmas : list of r (d,d) ndarrays or (r,d,d) ndarray
            Covariance matrices for each row of the operator matrix.
            That is, Sigmas[i] = Covariance(O[i])

        modelform : str
            Structure of the ROMs to sample.
        """
        self._init_means(means, modelform)

        # Factor the covariances for each operator row.
        self.covariances = np.array(Sigmas)
        self._chos = [np.linalg.cholesky(Σ) for Σ in Sigmas]

    def _init_means(self, means, modelform, m=1):
        """Record means and initialize dimensions."""
        # Check and save dimensions.
        self.means = np.array(means)
        r = means.shape[0]
        self._r, self._d = r, roi.lstsq.lstsq_size(modelform, r, m)
        assert self.means.shape == (self._r, self._d)

        # Get split indices for the operator matrix.
        if modelform == "cAHB":
            self._indices = np.cumsum([1, r, r*(r+1)//2])
        else:
            raise NotImplementedError(f"modelform='{modelform}'")
        self._modelform = modelform

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        rows = [µ + cho @ np.random.standard_normal(self._d)
                                for µ, cho in zip(self.means, self._chos)]
        return np.vstack(rows)

    def _construct_rom(self, O):
        """Construct the ROM from the operator matrix."""
        c_, A_, H_, B_ = np.split(O, self._indices, axis=1)
        rom = roi.InferredContinuousROM(self._modelform)
        return rom._set_operators(None, c_=c_.flatten(), A_=A_, H_=H_, B_=B_)

    @property
    def mean_rom(self):
        """Get the mean OpInf ROM from the posterior."""
        return self._construct_rom(self.means)

    def rvs(self):
        """Do a single sample of the posterior OpInf ROM distribution.

        Returns
        -------
        rom : roi.InferredContinuousROM
            A trained reduced-order model, representing a posterior draw.
        """
        return self._construct_rom(self._sample_operator_matrix())

    def predict(self, q0, t):
        """Draw a ROM from the posterior and simulate it from q0 over t."""
        return self.rvs().predict(q0, t, config.U, method="RK45")


class OpInfPosteriorUniformCov(OpInfPosterior):
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.

    This class is only for the special case in which the intial guess for λ
    is a single number (same penalization for each operator entry), resulting
    in the posterior distributions N(µi, σi^2 Σ) (i.e., same Σ for each i).
    """
    def __init__(self, means, sigmas, Sigma, modelform="cAHB"):
        """Construct the multivariate normal random variables.

        Parameters
        ----------
        means : (r,d) ndarray
            Mean values for each of the operator entries, i.e., E[O].

        sigmas : list of r floats or (r,) ndarray
            Scaling factors for each covariance matrix.

        Sigma : (d,d) ndarray
            Nonscaled covariance matrix for each posterior.
        """
        self._init_means(means, modelform)

        # Factor the covariances for each operator row.
        self._sigmas = sigmas
        self._Sigma = Sigma
        self._cho = np.linalg.cholesky(Sigma)

    @property
    def covariances(self):
        """Covariance matrices for each row of the operator matrix."""
        return np.array([(σ**2)*self._Sigma for σ in self._sigmas])

    def _sample_operator_matrix(self):
        """Sample an operator matrix from the posterior distribution."""
        rows = [µ + (σ*self._cho) @ np.random.standard_normal(size=self._d)
                                    for µ, σ in zip(self.means, self._sigmas)]
        return np.vstack(rows)


# Posterior construction ======================================================

def construct_posterior(trainsize, r, reg, case=2):
    """Construct the mean and covariance matrix for the posterior distribution,
    then create an object for sampling the posterior.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r : int
        The dimension of the ROM. This is also the number of retained POD modes
        (left singular vectors) used to project the training data.

    reg : float or (r,) ndarray or (r,d) ndarray
        The regularization parameter(s) used in the Operator Inference
        least-squares problem for training the ROM.
        * float: Λ = λi I
        * (r,) ndarray: Λi = λi I
        * (r,d) ndarray: Λi = diag(λi1,...,λid) (requires case=1)

    case : int
        How to treat the regularization update.
        * 2: learn a new λ for each ROW of the operator matrix.
        * 1: learn a new λ for each ENTRY of the operator matrix.

    Returns
    -------
    post : OpInfPosterior
        Posterior distribution object with rvs() sampling method.
    
    reg_new : (r,) ndarray or (r,d) ndarray
        The Bayesian update for the regularization parameters.
        * case == 2 -> (r,), a new λ for each ROW of the operator matrix.
        * case == 1 -> (r,d), a new λ for each ENTRY of the operator matrix.
    """
    # Get the data matrix and solve the Operator Inference problem,
    # using the initial guess for the regularization parameter(s).
    Q_, R, t, = utils.load_projected_data(trainsize, r)
    U = config.U(t).reshape((1,-1))
    rom = roi.InferredContinuousROM("cAHB").fit(None, Q_, R, U, reg)
    rom.trainsize = trainsize
    D = rom._assemble_data_matrix(Q_, U)
    O = rom.operator_matrix_

    # Check matrix shapes.
    d = 1 + r + r*(r+1)//2 + 1
    assert D.shape == (trainsize, d)
    assert R.shape == (r, trainsize)
    assert O.shape == (r, d)

    def symmetrize(S, sparsify=False):
        """Numerically symmetrize / sparsify (e.g., for covariance)."""
        S = (S + S.T) / 2
        if sparsify:
            S[np.abs(S) < 1e-16] = 0
        return S

    with utils.timed_block("Building posterior distribution"):
        # Precompute some quantities for posterior parameters.
        DTD = symmetrize(D.T @ D)
        Onorm2s = np.sum(O**2, axis=1)                  # ||o_i||^2.
        residual2s = np.sum((D @ O.T - R.T)**2, axis=0) # ||Do_i - r_i||^2.

#         print("||o_i||^2:", Onorm2s)
#         print(f"{Onorm2s.mean()} ± {Onorm2s.std()}")
#         print("||Do_i - r_i||^2:", residual2s)
#         print(f"{residual2s.mean()} ± {residual2s.std()}")
#         input("Press ENTER to continue")

        # Calculate posterior ROM distribution.
        if np.isscalar(reg):
            λ2 = reg**2
            Λ = λ2*np.eye(d)
            Σ = symmetrize(la.inv(DTD + Λ), sparsify=True)
            σ2s = (residual2s + λ2*Onorm2s) / trainsize
            post = OpInfPosteriorUniformCov(O, np.sqrt(σ2s), Σ, "cAHB")
            if case == 1:
                Σs = np.array([σ2i * Σ for σ2i in σ2s]) # = post.covariances
        else:
            λ2 = np.array(reg)**2
            if λ2.shape == (r,):
                I = np.eye(d)
                Λs = [λ2i*I for λ2i in λ2]
                σ2s = (residual2s + λ2*Onorm2s) / trainsize
            elif λ2.shape == (r,d):
                if case != 1:
                    raise ValueError("2D reg only compatible with case=1")
                Λs = [np.diag(λ2i) for λ2i in λ2]
                σ2s = (residual2s + np.sum(λ2*O**2, axis=1)) / trainsize
            else:
                raise ValueError("invalid shape(reg)")
            assert len(Λs) == len(σ2s) == r
            Σs = np.array([σ2i * symmetrize(la.inv(DTD + Λi), sparsify=True)
                                                for σ2i, Λi in zip(σ2s, Λs)])
            post = None
            # post = OpInfPosterior(O, Σs, modelform="cAHB")

    with utils.timed_block("Calculating updated regularization parameters"):
        if case == 2:  # So λ2 is a scalar or (r,) ndarray
            gs = la.eigvalsh(DTD)   # Non-negative eigenvalues of data Grammian.
            if np.isscalar(reg):    # Scalar regularization parameter
                gamma = np.sum(gs / (λ2 + gs))
            else:
                gamma = np.sum(gs / (λ2.reshape((-1,1)) + gs), axis=1)
                assert len(gamma) == r
            # print("Gamma exact:", gamma)
            # print("Gamma estimate (d):", d)
            λ2_new = gamma*σ2s/Onorm2s
        elif case == 1:
            # TODO: verify and fix. Note use of λ2_new, not λ_new.
            xi = np.zeros_like(O)
            badmask = np.ones_like(O, dtype=bool)
            pairs = []
            for i in range(O.shape[0]):
                for j in range(O.shape[1]):
                    pairs.append((Σs[i,j,j], O[i,j]**2))
                    s = Σs[i,j,j] / O[i,j]**2
                    if s < 1:
                        xi[i,j] = 1 - s + s**2 - s**3 + s**4 - s**5 + s**6
                        badmask[i,j] = False
            λ2_new = (xi / O**2) * σ2s.reshape((-1,1))
            λ2_new[badmask] = .01
            assert λ2_new.shape == (r,d)

            pairs = np.array(pairs)
            bad = (pairs[:,0] > pairs[:,1])
            plt.plot(pairs[~bad,0], pairs[~bad,1], 'C0.', ms=2, alpha=.2)
            plt.plot(pairs[ bad,0], pairs[ bad,1], 'C3.', ms=2, alpha=.2)
            plt.show()

        else:
            raise ValueError(f"invalid case ({case})")

    return post, np.sqrt(λ2_new)

# -----------------------------------------------------------------------------
def simulate_posterior(trainsize, post, ndraws=10, steps=None):
    """
    Parameters
    ----------
    rom : ...

    post : OpInfPosterior

    ndraws : int

    steps : int

    Returns
    -------
    q_rom_mean : (r,steps) ndarray
        TODO

    q_roms : list
        TODO

    scales : ndarray
        TODO
    """
    # Load the time domain and initial conditions.
    t = utils.load_time_domain(steps)
    q0 = utils.load_projected_data(trainsize, post._r)[0][:,0]

    # Simulate the mean ROM as a reference.
    with utils.timed_block(f"Simulating mean ROM"):
        q_rom_mean = post.mean_rom.predict(q0, t, config.U, method="RK45")

    # Get ndraws simulation samples.
    q_roms = []
    i = 0
    while i < ndraws:
        with utils.timed_block(f"Simulating posterior draw ROM {i+1}"):
            q_rom = post.predict(q0, t)
            if q_rom.shape[1] == t.shape[0]:
                q_roms.append(q_rom)
                i += 1

    return q_rom_mean, q_roms


# Main routines ===============================================================

def plot_mode_uncertainty(trainsize, mean, draws, modes=8):
    steps = mean.shape[1]
    t = utils.load_time_domain(steps)

    if len(draws) > 0:
        with utils.timed_block("Calculating sample deviations"):
            offsets = [draw - mean for draw in draws]
            deviations = np.std(offsets, axis=0)

    nrows = (modes//2) + 1 if modes % 2 else modes//2
    fig, axes = plt.subplots(nrows, 2)
    for i, ax in zip(range(modes), axes.flat):
        ax.plot(t, mean[i,:], 'C0-', lw=1)
        # for draw in draws:
        #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.2)
        if len(draws) > 0:
            ax.fill_between(t, mean[i,:] - 3*deviations[i,:],
                               mean[i,:] + 3*deviations[i,:], alpha=.5)
        if steps > trainsize:
            ax.axvline(t[trainsize], color='k', lw=1)
        # ax.set_title(fr"POD mode ${i+1}$")
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
    fig.tight_layout()


def plot_pointtrace_uncertainty(trainsize, mean, draws, var="p"):
    """

    Parameters
    ----------

    Returns
    -------
    """
    if var not in ["p", "vx", "vy", "T"]:
        raise NotImplementedError(f"var='{var}'")

    # Get the indicies for each variable.
    elems = np.atleast_1d(config.MONITOR_LOCATIONS)
    nelems = elems.size
    nrows = (nelems // 2) + (1 if nelems % 2 != 0 else 0)
    elems = elems + config.ROM_VARIABLES.index(var)*config.DOF

    # Load the true pressure traces and the time domain.
    traces_gems, t = utils.load_gems_data(rows=elems)
    steps = mean.shape[1]
    t = t[:steps]
    traces_gems = traces_gems[:,:steps]

    # Load the basis rows corresponding to the pressure traces.
    V, scales = utils.load_basis(trainsize, mean.shape[0])
    Velems = V[elems]

    # Reconstruct and rescale the simulation results.
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom_mean = dproc.unscale(Velems @ mean, scales, var)
        traces_rom_draws = [dproc.unscale(Velems @ draw, scales, var)
                            for draw in draws]

    with utils.timed_block("Calculating sample deviations"):
        offsets = [draw - traces_rom_mean for draw in traces_rom_draws]
        deviations = np.std(offsets, axis=0)

    fig, axes = plt.subplots(nrows, 2, figsize=(9,6), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, traces_gems[i,:], lw=1, **config.GEMS_STYLE)
        ax.plot(t, traces_rom_mean[i,:], 'C0-', lw=1,
                label=r"ROM ($\mu$)")
        # for draw in traces_rom_draws:
        #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.25)
        ax.fill_between(t, traces_rom_mean[i,:] - 3*deviations[i,:],
                           traces_rom_mean[i,:] + 3*deviations[i,:], alpha=.5)
        ax.axvline(t[trainsize], color='k', lw=1)
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_title(f"Location ${i+1}$", fontsize=12)
        ax.locator_params(axis='y', nbins=2)
    for ax in axes[-1,:]:
        ax.set_xlabel("Time [s]", fontsize=12)
    for ax in axes[:,0]:
        ax.set_ylabel(config.VARLABELS[var], fontsize=12)

    # Single legend to the right of the subplots.
    fig.tight_layout(rect=[0, 0, .85, 1])
    leg = axes[0,0].legend(loc="center right", fontsize=14,
                           bbox_to_anchor=(1,.5),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)


def main(trainsize, r, reg, ndraws=10, steps=50000, modes=4):
    post = construct_posterior(trainsize, r, reg, case=2)[0]
    mean, draws = simulate_posterior(trainsize, post, ndraws, steps)
    plot_mode_uncertainty(trainsize, mean, draws, modes)
    utils.save_figure("bayes/bayes_first4modes.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, var="p")
    utils.save_figure("bayes/bayes_traces_pressure.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, var="T")
    utils.save_figure("bayes/bayes_traces_temperature.pdf")


def iterate(trainsize, r, reg, niter, case=2):
    """Do the iteration several times, plotting the evolution thereof."""
    print(f"Initialization: reg = {reg}")
    means = np.empty(niter+1, dtype=float)
    stds = means.copy()
    means[0], stds[0] = reg**2, 0
    iterations = np.arange(niter+1)
    for n in iterations[1:]:
        post, reg = construct_posterior(trainsize, r, reg, case=case)
        print(f"Iteration {n}: reg = {reg}")
        means[n], stds[n] = np.mean(reg**2), np.std(reg**2)
    print("Relative change in mean at final update:",
          f"{abs(means[-1] - means[-2]) / abs(means[-1]):%}")

    # Plot progression of regularization statistics.
    plt.semilogy(iterations, means, 'C0.-', ms=10)
    plt.fill_between(iterations, means-stds, means+stds,
                     color="C0", alpha=.5)
    plt.xlabel("Bayes Iteration")
    plt.ylabel(r"Regularization $\lambda$ ($\mu \pm \sigma$)")
    plt.title("Iterative Bayesian Regularization Update: Combustion")
    plt.xlim(right=niter)
    utils.save_figure(f"bayes/iteration_case{case}.pdf")

    # Try simulating the final model.
    mean, draws = simulate_posterior(trainsize, post, 0, trainsize)
    plot_mode_uncertainty(trainsize, mean, draws, 4)
    utils.save_figure(f"bayes/iter{case}_first4modes.pdf")



if __name__ == "__main__":
    # main(20000, 40, 36382, 100, 50000, 4)
    iterate(20000, 40, 36382, 15, case=1)
