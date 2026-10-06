"""
AMOEBA buffered 14-7 van der Waals energy (pure JAX).

Halgren's buffered 14-7 potential [Halgren1992]_ as used by AMOEBA
[Ponder2010]_,

.. math::

    E_{ij} = \\epsilon_{ij} \\left(\\frac{1.07}{\\rho + 0.07}\\right)^7
             \\left(\\frac{1.12}{\\rho^7 + 0.12} - 2\\right),
    \\qquad \\rho = r_{ij} / \\sigma_{ij},

evaluated between *reduced* interaction sites: a hydrogen with parent ``p``
and reduction factor ``f`` interacts from ``x_p + f (x_H - x_p)``.  The
optional ``LennardJones`` potential function of ``openmm.AmoebaVdwForce`` is
also supported.  Pair parameters, exclusions and per-particle scale factors
are precomputed in :func:`~mars.forcefields.amoeba.convert.vdw_from_openmm`.

Ground truth is ``AmoebaReferenceVdwForce.cpp`` (OpenMM Reference platform).

References
----------
.. [Halgren1992] T. A. Halgren, J. Am. Chem. Soc. 114, 7827 (1992).
.. [Ponder2010] J. W. Ponder et al., J. Phys. Chem. B 114, 2549 (2010).
"""

import jax.numpy as jnp

from .convert import VdwParams


def reduced_sites(pos_nm, params: VdwParams):
    """Interaction sites after hydrogen reduction, ``(N, 3)`` in nm."""
    red = jnp.asarray(params.reduction)[:, None]
    par = pos_nm[jnp.asarray(params.parent)]
    reduced = par + red * (pos_nm - par)
    return jnp.where(red != 0.0, reduced, pos_nm)


def vdw_energy(pos_nm, params: VdwParams):
    """AMOEBA vdW energy (kJ/mol) for positions in nm.

    Parameters
    ----------
    pos_nm : (N, 3) array
    params : VdwParams

    Returns
    -------
    scalar
    """
    if params.pair_i.size == 0:
        return jnp.zeros((), dtype=pos_nm.dtype)
    sites = reduced_sites(pos_nm, params)
    d = sites[jnp.asarray(params.pair_j)] - sites[jnp.asarray(params.pair_i)]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    sigma = jnp.asarray(params.sigma)
    eps = jnp.asarray(params.epsilon)
    if params.lennard_jones:
        p6 = (sigma / r) ** 6
        return jnp.sum(4.0 * eps * (p6 * p6 - p6))
    rho = r / sigma
    rho7 = rho**7
    t1 = (1.07 / (rho + 0.07)) ** 7
    t2 = 1.12 / (rho7 + 0.12)
    return jnp.sum(eps * t1 * (t2 - 2.0))
