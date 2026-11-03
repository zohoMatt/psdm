# The model

This page describes what `psdm` computes. `psdm` is a restructured port of the US EPA Python
PSDM ([Water_Treatment_Models](https://github.com/USEPA/Water_Treatment_Models/tree/3f514c2/PSDM),
commit `3f514c2`), which was converted from the Fortran PSDM in AdDesignS™. The numerics follow
the EPA code; where a statement below rests only on the EPA code (and not on a primary
publication we have checked), it says so.

## Physical picture

Water flows through a packed bed of porous adsorbent particles. Each dissolved compound

1. is carried along the bed by advection (axial dispersion is neglected),
2. crosses the liquid film around each particle (film coefficient $k_f$),
3. diffuses into the particle through the pore liquid (pore diffusivity $D_p$) and along the
   internal surface (surface diffusivity $D_s$), and
4. is in local equilibrium between pore liquid and adsorbed phase.

Single-solute equilibrium is the Freundlich isotherm $q = K C^{1/n}$. With several compounds,
competition is described by ideal adsorbed solution theory (IAST) for Freundlich isotherms.

## Equations

In dimensional form, for each compound in a spherical particle of radius $R$, with particle
porosity $\varepsilon_p$, apparent particle density $\rho_p$, pore-liquid concentration $C_p$ and
adsorbed-phase concentration $q$:

$$
\varepsilon_p \frac{\partial C_p}{\partial t} + \rho_p \frac{\partial q}{\partial t}
= \frac{1}{r^2}\frac{\partial}{\partial r}\left[ r^2 \left(
\varepsilon_p D_p \frac{\partial C_p}{\partial r} + \rho_p D_s \frac{\partial q}{\partial r}
\right)\right],
$$

with symmetry at $r = 0$ and a film-transfer condition at $r = R$:

$$
\left.\varepsilon_p D_p \frac{\partial C_p}{\partial r} + \rho_p D_s \frac{\partial q}{\partial r}
\right|_{r=R} = k_f \left( C_b - C_p|_{r=R} \right).
$$

The bulk liquid in a bed of void fraction $\varepsilon$ and superficial velocity $v$ satisfies

$$
\varepsilon \frac{\partial C_b}{\partial t} + v \frac{\partial C_b}{\partial z}
= -\frac{3 (1-\varepsilon)}{R} k_f \left( C_b - C_p|_{r=R} \right),
$$

with the influent concentration prescribed at $z = 0$ and $q$, $C_p$, $C_b$ initially zero.

The solver works with these equations in dimensionless form, using the solute distribution
parameters (surface and pore), the surface and pore diffusion moduli, and a Stanton number, as
in the EPA code (`psdm.solver._build_system` documents each group next to its definition).
Time is scaled by the fluid residence time multiplied by $1 + \sum D_g$.

### Fouling

With a background water other than `"Organic Free"`, the Freundlich $K$ of each compound
decreases with time as

$$
\frac{K(t)}{K_0} = (b_1 a_1 + b_2) + b_1 a_2\, t + b_1 a_3\, e^{b_1 a_4 t}
\quad (t \text{ in minutes, floored at } 10^{-3}),
$$

where $a_1 \ldots a_4$ depend on the water and $b_1, b_2$ on the compound class (tables in
{mod}`psdm.fouling`), taken from the AdDesignS database files `water_co.txt` and
`corr_com.txt`. The floor of $10^{-3}$ is the `XKTIME` floor of the AdDesignS `DIFFUN`.

**Tortuosity correlation.** AdDesignS offers a time-dependent tortuosity correlation, on by
default in fouled water here (`tortuosity_correlation=`). While it is on, the tortuosity used
for $D_p$ and $D_s$ is 1, user values of $D_p$ and $D_s$ are not accepted, surface diffusion is
switched off, and the pore-diffusion term is multiplied by

$$
F = \frac{1/\tau(t) - D}{1 - D}, \qquad
\tau(t) = \begin{cases} 1 & t \le 1.008\times10^5 \text{ min (70 d)} \\
0.334 + 6.61\times10^{-6}\, t & \text{otherwise} \end{cases}
$$

with $D = D_s/D_p$ (AdDesignS `DIFFUN`, `FAC`). AdDesignS switches surface diffusion off by
forcing the SPDFR to $10^{-30}$. That makes $D_s \approx 10^{-40}$ cm²/s, where scipy's BDF
integrator does not converge, so psdm divides $D_s$ by $10^{10}$ instead, as the EPA code
does. Surface diffusion is negligible either way, and the curves match the AdDesignS output.

## Numerical method

Both coordinates are discretised by orthogonal collocation: symmetric polynomials in $r^2$ at
the roots of a Jacobi polynomial plus the particle surface (`nr` points), and shifted Legendre
roots plus both ends along the bed (`nz` points per element, optionally on `ne` finite
elements with first-derivative continuity between elements). The resulting stiff ODE system is
integrated with {func}`scipy.integrate.solve_ivp` (BDF by default, `rtol=1e-3`, `atol=1e-6`)
using a Jacobian sparsity pattern derived from the structure of the equations (checked in the
test suite against a finite-difference Jacobian).

## Parameter correlations

- **Water viscosity and density** as functions of temperature: empirical fits taken from the
  EPA code (no source given there). At 20 °C the viscosity fit gives 1.018 cP, about 1.6 %
  above the tabulated 1.002 cP.
- **Liquid diffusivity**: Hayduk & Laudie (1974),
  $D_l = 13.26\times10^{-5} / (\mu^{1.14} V_b^{0.589})$ with $\mu$ in cP and $V_b$ in cm³/mol.
- **Pore diffusivity**: $D_p = D_l / \tau$, or user-supplied. The pore diffusion modulus is
  $E_{dp} = D_p D_{gp} \tau_f / R^2$ as in AdDesignS (the EPA code multiplies it by $\tau$
  again, which differs from AdDesignS when $\tau \ne 1$).
- **Surface diffusivity**: from the surface-to-pore diffusion flux ratio $S$ (parameter
  `psdfr`, the name used in the EPA code):
  $D_s = S\, \varepsilon_p D_p C_0 / (\rho_p q_0)$, where $q_0$ is the equilibrium loading at
  the influent concentration $C_0$; or user-supplied.
- **Film transfer** $k_f$: the AdDesignS liquid-phase correlation (labelled "Gnielinski" in
  AdDesignS and "Chern and Chien" in the EPA code),
  $k_f = SF\,[1 + 1.5(1-\varepsilon)]\,(D_l/2R)\,(2 + 0.644\,Re^{1/2} Sc^{1/3})$, with the
  particle shape factor $SF$ (`Column.shape_factor`, default 1).

## References

Checked against Crossref metadata:

- Hand, D. W., Crittenden, J. C. & Thacker, W. E. (1984). Simplified models for design of
  fixed-bed adsorption systems. *J. Environ. Eng.* 110(2), 440–456.
  doi:[10.1061/(ASCE)0733-9372(1984)110:2(440)](https://doi.org/10.1061/(ASCE)0733-9372(1984)110:2(440)).
  Background on dimensionless fixed-bed models (homogeneous surface diffusion model).
- Crittenden, J. C. et al. (1985). Prediction of multicomponent adsorption equilibria using
  ideal adsorbed solution theory. *Environ. Sci. Technol.* 19(11), 1037–1043.
  doi:[10.1021/es00141a002](https://doi.org/10.1021/es00141a002). IAST with Freundlich isotherms.
- Hayduk, W. & Laudie, H. (1974). Prediction of diffusion coefficients for nonelectrolytes in
  dilute aqueous solutions. *AIChE J.* 20(3), 611–615.
  doi:[10.1002/aic.690200329](https://doi.org/10.1002/aic.690200329).
- Burkhardt, J. B., Burns, N., Mobley, D., Pressman, J. G., Magnuson, M. L. & Speth, T. F.
  (2022). Modeling PFAS removal using granular activated carbon for full-scale system design.
  *J. Environ. Eng.* 148(3), 04021086.
  doi:[10.1061/(ASCE)EE.1943-7870.0001964](https://doi.org/10.1061/(ASCE)EE.1943-7870.0001964).
  Application of the EPA PSDM to PFAS pilot data.

Software:

- Hokanson, D. R., Hand, D. W., Crittenden, J. C., Rogers, T. N. & Oman, E. J. AdDesignS™ —
  Adsorption Design Software, Michigan Technological University.
- US EPA, Water Treatment Models, PSDM:
  <https://github.com/USEPA/Water_Treatment_Models/tree/3f514c2/PSDM>.
