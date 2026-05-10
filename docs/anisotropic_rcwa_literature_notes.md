# Anisotropic RCWA Literature Notes

These notes map the local anisotropic-RCWA reference papers to the current
anisotropic solver implementation.  They are intended as an engineering checklist:
each algorithmic step should have a paper trail before it is promoted into the
core solver.

## Core References Read

- D. W. Berreman, "Optics in Stratified and Anisotropic Media: 4 x 4-Matrix
  Formulation", JOSA 62, 502-510 (1972).
- L. Li, "Reformulation of the Fourier modal method for surface-relief gratings
  made with anisotropic materials", Journal of Modern Optics 45, 1313-1334
  (1998).
- L. Li, "Fourier modal method for crossed anisotropic gratings with arbitrary
  permittivity and permeability tensors", Journal of Optics A 5, 345-355
  (2003).
- K. Watanabe, R. Petit, and M. Neviere, "Differential theory of gratings made
  of anisotropic materials", JOSA A 19, 325-334 (2002).
- I. Smagin, S. Dyakov, and N. Gippius, "The Fourier modal method for gratings
  with bi-anisotropic materials", arXiv:2510.05973 (2025).
- The local anisotropic-RCWA reference-list DOCX, which places the above papers in the broader
  RCWA/FMM lineage.

## What The Current Code Already Implements

1. **Berreman-style 4-component state.**  The solver evolves the tangential
   fields `[Ex, Ey, Hx, Hy]` and eliminates longitudinal components through
   Maxwell constraints.  This is the same state philosophy as Berreman's 4 x 4
   stratified-anisotropic formulation.

   Current locations:
   - `src/rcwa3d_anisotropic/solver.py::homogeneousOrderSystemMatricesBackend`
   - `src/rcwa3d_anisotropic/solver.py::liFactorizedSystemMatrixBackend`

2. **Layer eigenproblem plus S-matrix cascading.**  Each vertically homogeneous
   Fourier layer is converted to an eigenproblem, then connected through
   interface continuity and Redheffer/S-matrix cascading.  This matches the
   FMM/RCWA architecture described in Li 1998/2003 and the 2025 bi-anisotropic
   FMM paper.

   Current locations:
   - `src/rcwa3d_anisotropic/solver.py`
   - `src/rcwa3d_anisotropic/smatrix.py`

3. **Electric tensor convolution with z-normal Li inverse rule.**  The code forms
   convolution matrices for the permittivity tensor and uses
   `[epsilon_zz]^-1` to eliminate `Ez`.  This is a partial Li-factorized
   anisotropic formulation.  It is especially close to the non-magnetic,
   no-magnetoelectric limit of the 2025 formulation.

   Current locations:
   - `src/rcwa3d_anisotropic/factorization.py::tensorConvolutionData`
   - `src/rcwa3d_anisotropic/solver.py::liFactorizedSystemMatrixBackend`

4. **Normal-vector Li factorization for scalar discontinuous shapes.**  For
   scalar patterned layers, the code can decompose fields into local normal and
   tangential directions, using an inverse rule in the normal direction and a
   direct rule tangentially.  This follows the NVF line in Schuster/Gotz and is
   consistent with the Fourier-factorization problem highlighted by Li and
   Watanabe.

   Current locations:
   - `src/rcwa3d_anisotropic/factorization.py::normalVectorScalarTensorMatrices`
   - `src/rcwa3d_anisotropic/factorization.py::analyticNormalVectorScalarTensorMatrices`

5. **Analytic Fourier geometry for standard scalar shapes.**  Rectangles now use
   sinc coefficients; disks and ellipses use the `2 J1(x) / x` coefficient.  This
   avoids staircase sampling for simple shapes, matching the design pattern used
   by mature Fourier-modal software such as S4.

   Current locations:
   - `src/rcwa3d_anisotropic/analytic.py`
   - `src/rcwa3d_anisotropic/geometry.py`

6. **Homogeneous electric-magnetic tensor layers.**  The material data model now
   accepts `ConstitutiveTensors(epsilon, mu, chi, xi)` and `Layer(..., mu=...)`.
   The implemented solver path covers homogeneous constant `epsilon` and
   constant `mu` with `chi = xi = 0`.  It eliminates both `Ez` and `Hz`, using
   `Dz = Ky Hx - Kx Hy` and `Bz = Kx Ey - Ky Ex`, then builds the Berreman-style
   `[Ex, Ey, Hx, Hy]` first-order matrix.  This is the homogeneous-layer limit of
   the electric-magnetic tensor formulation in Li 2003.  Tests check that
   `mu = I` is a numerical regression to the previous implementation.

   Current locations:
   - `src/rcwa3d_anisotropic/constitutive.py`
   - `src/rcwa3d_anisotropic/solver.py::homogeneousOrderSystemMatricesBackend`
   - `tests/testSolverPaths.py::testAnisotropicMuIdentityMatchesExistingHomogeneousTensorPath`

## Important Gaps Against The Literature

1. **Li 2003 patterned arbitrary epsilon and mu tensors are not fully
   implemented.**  Homogeneous constant `epsilon/mu` layers are now supported,
   but patterned layers still assume relative permeability is identity unless
   the supplied `mu` is exactly `I`.  Full Li 2003 crossed-grating support
   requires Fourier convolution/factorization for the magnetic tensor blocks,
   not only the homogeneous `Hz` Schur complement.

2. **Bi-anisotropic magnetoelectric coupling is not implemented.**  The 2025
   paper uses constitutive relations
   `D = epsilon E + chi H`, `B = xi E + mu H`.  Current code exposes `chi` and
   `xi` in the material model but rejects nonzero values explicitly.  Supporting
   chirality/Tellegen materials needs coupled Fourier tensors for `epsilon`,
   `chi`, `xi`, and `mu`, and cannot be bolted on as a small parameter in the
   existing `epsilon` mapping.

3. **The tensor Fourier factorization is partial.**  Current tensor layers use
   direct component convolution plus a z-normal inverse rule for `epsilon_zz`.
   Li 1998/2003 derive more complete operator combinations (`Q`-type or
   `L1/L2`-type operators) for discontinuous anisotropic tensors.  The current
   implementation is reasonable for many structures but is not yet the full Li
   anisotropic FFF.

4. **Freeform tensor interfaces do not yet have a full anisotropic NVF or
   complex-polarization basis.**  The implemented normal-vector route applies to
   scalar patterns.  For discontinuous tensor-valued patterns, Antos/Veis-style
   complex polarization bases or a generalized NVF tensor factorization would be
   needed.

5. **Profiled surfaces are still sliced.**  Watanabe 2002 emphasizes avoiding
   staircase approximation of grating profiles in differential theory.  The
   current layer stack handles z-varying geometry by slicing.  This is standard
   RCWA practice but not the same as Watanabe's continuous-profile differential
   formulation.

6. **Mode classification is numerical rather than Booker-quartic analytic.**
   Li 1998 gives explicit criteria for upward/downward homogeneous anisotropic
   plane waves.  Current code sorts modes by flux/decay heuristics after the
   eigenproblem.  This is practical for dense RCWA, but near degeneracies and
   optic-axis singularities should be tested against Li's criteria.

## Recommended Deep-Improvement Roadmap

1. **Extend the formal material-tensor model.**  `ConstitutiveTensors` now
   exists and preserves the current `Layer.epsilon` compatibility shortcut.
   The remaining work is to route this model through patterned tensor
   factorization and field reconstruction for every material tensor.

2. **Complete the Li 2003 electric-magnetic tensor path.**  The homogeneous
   `epsilon/mu` path is implemented and validates that `mu = I` reduces to the
   previous solver.  Next add patterned magnetic tensor convolution/factorization
   and benchmark cases from Li 2003.

3. **Implement the 2025 bi-anisotropic Scheme 1.**  Scheme 1 uses Laurent-rule
   Fourier tensors for `epsilon`, `mu`, `chi`, and `xi`.  It is algebraically
   simpler and is a good stepping stone before full Scheme 2 factorization.

4. **Implement the 2025 bi-anisotropic Scheme 2 only after Scheme 1 tests pass.**
   Scheme 2 couples all four tensors in the factorization.  It should be gated
   behind explicit tests because an incorrect implementation can look plausible
   while violating energy balance or convergence behavior.

5. **Add convergence benchmark tests from papers.**  Use Li 1998/2003 numerical
   tables where possible, plus a scalar limit test against the isotropic solver.
   Every new tensor formulation should have:
   - homogeneous-layer reduction test,
   - scalar/isotropic limit test,
   - diagonal anisotropic limit test,
   - nonzero off-diagonal tensor test,
   - Fourier-order convergence snapshot.

6. **Improve mode sorting near degeneracy.**  Add diagnostic tests around
   grazing/Rayleigh anomalies and optic-axis-like degeneracies.  Compare with
   Li 1998's upward/downward wave criteria for homogeneous anisotropic media.

## Practical Guidance For Current Examples

- For Si/InAs/Ag nonreciprocal thermal emitter examples, the present solver is in
  the non-magnetic patterned anisotropic RCWA class: gyrotropic permittivity is
  allowed, but the patterned-layer path still uses `mu = I` and `chi = xi = 0`.
  Homogeneous spacer or film layers may use a constant `mu` tensor.
- Use `analytic=True` for rectangular, circular, and elliptical scalar geometry
  whenever possible.  It removes staircase geometry error and makes convergence
  studies cleaner.
- Keep `factorization="auto"` for scalar high-contrast patterned layers.  Use
  `factorization="standard"` only for debugging and literature comparisons.
- For tensor-valued discontinuous patterns, current results should be treated as
  a partial Li-factorized formulation until full Li 2003 tensor FFF is added.
