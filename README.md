# Topological Data Analysis for Two Species Diffusion Models

The multi-species exclusion process (MSEP) on a one-dimensional ring is an interacting particle system in which particles of different species exchange positions according to prescribed transition rates. From this process, one can generate directed polymer configurations whose large-scale fluctuations are connected to the Kardar-Parisi-Zhang (KPZ) universality class, which describes the growth of rough interfaces. Using nonlinear fluctuating hydrodynamics (NLFH), the MSEP can be decomposed into hydrodynamic normal modes, with the number of independent modes equal to one less than the number of particle species. These modes may relax according to different universality classes, such as KPZ or diffusive scaling. In this project, we apply topological data analysis (TDA) to point clouds generated from time-varying projections of the directed polymer configurations into lower-dimensional space. Persistent homology is then used to measure how the topology of these projected configurations changes and relaxes over time. By studying how the characteristic relaxation time of these topological signatures scales with system size, we estimate the dynamical critical exponent of the process. This provides a geometric approach to detecting hydrodynamic relaxation and identifying universal scaling behavior in multi-species exclusion processes.

## Accomplishments

- Simulated multi-species diffusion process given the swapping rates, length, number of species (dimension) and density of species.
- Successfully extracted roughness exponent ($\alpha$), growth exponent ($\beta$) and the dynamical critical exponent ($z$) critical exponents from multi-species diffusion processes'.
- Extracted dynamical critical exponent from TDA of normal mode time series

## Current Goals

- Try to extract dynamical critical exponent from raw MSEP process.

## References

- Schütz, G. M., & Wehefritz–Kaufmann, B. (2017). [Kardar-Parisi-Zhang modes in d-dimensional directed polymers.](https://www.math.purdue.edu/~ebkaufma/PhysRevE.96.032119.pdf) Physical Review E, 96(3), 032119.
- Wehefritz-Kaufmann, B. (2010). [Dynamical critical exponent for two-species totally asymmetric diffusion on a ring.](https://sigma-journal.com/2010/039/sigma10-039.pdf) SIGMA. Symmetry, Integrability and Geometry: Methods and Applications, 6, 039.
- Chazal, F., & Michel, B. (2021). [An introduction to topological data analysis: fundamental and practical aspects for data scientists.](https://arxiv.org/pdf/1710.04019) Frontiers in artificial intelligence, 4, 667963.
- Krebs, K., Pfannmüller, M. P., Wehefritz, B., & Hinrichsen, H. (1995). [Finite-size scaling studies of one-dimensional reaction-diffusion systems](https://link.springer.com/content/pdf/10.1007/BF02180139.pdf). Part I. Analytical results. Journal of statistical physics, 78(5), 1429-1470.
- Popkov, V., Schadschneider, A., Schmidt, J., & Schütz, G. M. (2016). [Exact scaling solution of the mode coupling equations for non-linear fluctuating hydrodynamics in one dimension. Journal of statistical mechanics: theory and experiment](https://arxiv.org/pdf/1608.03267), 2016(9), 093211.

This work was created by [Esha Sury](mailto:esury@purdue.edu) and [Dhruv Upreti](mailto:dupreti@purdue.edu). Mentorship was given by [Birgit Kaufmann](mailto:ebkaufma@purdue.edu).
