# The model

## Physical picture

Water flows through a packed bed of porous adsorbent particles. Each dissolved compound

1. is carried along the bed by advection (dispersion is neglected),
2. crosses the liquid film around each particle (film coefficient $k_f$),
3. diffuses into the particle through the pore liquid (pore diffusivity $D_p$) and along the
   internal surface (surface diffusivity $D_s$), and
4. is in local equilibrium between pore liquid and adsorbed phase.

Equilibrium is described by the Freundlich isotherm $q = K C^{1/n}$. With several compounds,
competition follows ideal adsorbed solution theory (IAST).

## Equations

Schematically, in dimensionless form, with $r$ the particle radius coordinate and $z$ the bed coordinate,
the particle phase satisfies

$$
\frac{\partial q}{\partial t} = \frac{1}{r^2}\frac{\partial}{\partial r}
\left[ r^2 \left( E_{ds} \frac{\partial q}{\partial r}
+ E_{dp} \frac{\partial C_p}{\partial r} \right) \right]
$$

with a film-transfer boundary condition at $r = 1$, and the bulk liquid satisfies

$$
\frac{\partial C_b}{\partial t} = -D_{g}\frac{\partial C_b}{\partial z}
- 3\, St\, (C_b - C_{p}|_{r=1}).
$$

$E_{ds}$, $E_{dp}$ (surface and pore diffusion moduli), the solute distribution parameters
$D_g$ and the Stanton number $St$ follow Hand et al. (1984) and the AdDesignS manual.

## Numerical method

Both coordinates are discretised by orthogonal collocation: symmetric polynomials in $r^2$ at
Jacobi roots in the particle (`nr` points including the surface), and shifted Legendre roots
along the bed (`nz` points per element, optionally on `ne` finite elements). The resulting
stiff ODE system is integrated with {func}`scipy.integrate.solve_ivp` (BDF by default) using
an analytically known Jacobian sparsity pattern.

## Correlations

- Water viscosity and density: AdDesignS polynomial fits in temperature.
- Liquid diffusivity: Hayduk & Laudie (1974).
- Film transfer: Chern & Chien (2002) by default, others in {data}`psdm.correlations.SHERWOOD`.
- Surface diffusivity: from the pore-to-surface diffusion flux ratio (`psdfr`), as in AdDesignS.
- Fouling: AdDesignS water/compound-class parameters; PFAS values from Burkhardt et al. (2022).

## References

- Hand, D. W., Crittenden, J. C. & Thacker, W. E. (1984). Simplified models for design of fixed-bed
  adsorption systems. *J. Environ. Eng.* 110(2), 440–456.
- Crittenden, J. C., Luft, P., Hand, D. W., Oravitz, J. L., Loper, S. W. & Ari, M. (1985).
  Prediction of multicomponent adsorption equilibria using ideal adsorbed solution theory.
  *Environ. Sci. Technol.* 19(11), 1037–1043.
- Hokanson, D. R., Hand, D. W., Crittenden, J. C., Rogers, T. N. & Oman, E. J. AdDesignS™ —
  Adsorption Design Software, Michigan Technological University.
- Burkhardt, J. B. et al. (2022). Modeling PFAS removal using granular activated carbon for
  full-scale system design. *J. Environ. Eng.* 148(3).
- US EPA, Water Treatment Models: <https://github.com/USEPA/Water_Treatment_Models>
