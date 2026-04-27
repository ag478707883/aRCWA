# Algorithm Notes

## Source layout

The numerical core is split by algorithm responsibility:

- `src/rcwa3d_isotropic/fourier.py`: Fourier order enumeration, normalized `Kx/Ky`
  construction, and scalar permittivity convolution matrices. This is where
  rectangular periods `period_x != period_y` and rectangular/circular harmonic
  truncation live.
- `src/rcwa3d_isotropic/phase.py`: square-root branch selection, forward `kz` convention,
  s/p plane-wave tangential fields, order-vector packing, and Poynting flux.
- `src/rcwa3d_isotropic/analytic.py`: closed-form scalar convolution matrices
  for disks, ellipses, rectangles, and annuli. These avoid sampled staircasing
  when geometry helpers are called with `analytic=True`.
- `src/rcwa3d_isotropic/smatrix.py`: stable scattering-matrix interface, propagation,
  and Redheffer star-product cascading. This is the default solver path and is
  the same stability idea used by mature RCWA/FMM reference codes.
- `src/rcwa3d_isotropic/solver.py`: the consolidated isotropic numerical core:
  material factorization, homogeneous bases, layer modes, prepared stacks,
  S-matrix cascading, diffraction-order power, and PyTorch/CUDA evaluation.
  The isotropic solver now exposes only the S-matrix
  method; `solveStackBatch` still reuses one prepared stack for TE/TM/custom
  incident amplitudes.
- `src/rcwa3d_isotropic/builder.py`: mutable patterned layer builder used by
  `RCWASimulation.addLayer(..., shape=(ny, nx))`.
- `src/rcwa3d_isotropic/simulation.py`: high-level isotropic wrapper matching
  the anisotropic API style. Static layers are precompiled once, spectra reuse
  batch solves, prepared stacks are cached with an LRU cache, and the same
  `polarization="TE"/"TM"` names are accepted.
- `src/rcwa3d_anisotropic/__init__.py`: reserved public entry point for the
  tensor-material solver, intentionally kept separate from the isotropic path.
  The anisotropic implementation uses `factorization.py` for tensor Fourier
  convolution/Li factorization and `solver.py` for the full first-order
  eigenproblem and S-matrix cascade.

This first implementation solves isotropic, non-magnetic, 2D-periodic layers
stacked along z. Coordinates are normalized by `k0 = 2*pi / wavelength`, and
the tangential Fourier coefficients are stored as:

```text
f = [Ex, Ey, Hx, Hy]^T
```

For each diffraction order:

```text
Kx = kx / k0 = n_inc sin(theta) cos(phi) + mx wavelength / period_x
Ky = ky / k0 = n_inc sin(theta) sin(phi) + my wavelength / period_y
```

The harmonic set can be rectangular or circular. Rectangular truncation keeps
all `mx=-Nx..Nx`, `my=-Ny..Ny` orders. Circular truncation keeps only orders in
the scaled reciprocal-space disk:

```text
(mx / Nx)^2 + (my / Ny)^2 <= 1
```

For `Nx != Ny` this is elliptical in integer-order space, which is the scaled
circular k-vector domain commonly used for rectangular 2D unit cells. The
isotropic and anisotropic public examples both use circular truncation by
default because it usually gives better accuracy per retained Fourier mode for
2D metasurfaces.

For an isotropic layer with relative permittivity convolution matrix `E`, the
state equation is:

```text
d/d(k0 z) f = i A f
```

with block matrices:

```text
[Ex']   [ P11 P12 ] [Hx]
[Ey'] = [ P21 P22 ] [Hy]

[Hx']   [ Q11 Q12 ] [Ex]
[Hy'] = [ Q21 Q22 ] [Ey]
```

where:

```text
P11 = Kx E^-1 Ky
P12 = I - Kx E^-1 Kx
P21 = Ky E^-1 Ky - I
P22 = -Ky E^-1 Kx

Q11 = -Kx Ky
Q12 = Kx Kx - E
Q21 = E - Ky Ky
Q22 = Ky Kx
```

The layer modes are eigenvectors of `A`. The field inside each finite layer is:

```text
f_j(z) = V_j diag(exp(i q_j k0 (z - z_j))) c_j
```

The default solver uses interface scattering matrices, stable layer propagation
matrices, and Redheffer star-product cascading. Propagation through a finite
layer uses only `exp(i q k0 d)` for the forward-branch eigenvalues, so evanescent
orders decay instead of appearing as exponentially growing terms.

Use `solveStack(..., backend="cuda")` or `solveStackBatch(..., backend="cuda")`
to run the modal solve and S-matrix cascade on GPU via PyTorch. The isotropic
public solve path is CUDA-only: `backend=None`, `"auto"`, `"torch"`, and
`"gpu"` all resolve to CUDA and raise if no CUDA device is visible. CPU and
torch-cpu requests are rejected instead of falling back silently.

For homogeneous scalar layers the isotropic solver skips the dense RCWA
eigenproblem and builds the layer modes directly from the homogeneous s/p
plane-wave basis for every retained diffraction order. This is the default
fast path for uniform spacers, films, and sampled grids that are numerically
constant.

All isotropic examples use the same public numerical path: define materials,
build a complete layer stack, pattern selected `x/y/z` regions, compile the
layers once for the requested harmonic set, then call `solveStack` for each
wavelength, angle, and incident polarization:

```python
geometry = rcwa.LayerStack(period=(3.5, 1.0), shape=(8, 512))
geometry.addLayer(0.7, 1.0, name="air layer patterned with silicon", factorization="standard")
geometry.addLayer(0.1, -640.0 + 140.0j, name="gold film", factorization="standard")
geometry.setMaterial(3.4**2, x=(-0.13, 0.13), y=(-0.5, 0.5), z=(0.0, 0.7))

compiled_layers = rcwa.compileLayers(geometry.toLayers(), orders=(10, 0), truncation="circular")

result = rcwa.solveStack(
    layers=compiled_layers,
    wavelength=4.132,
    period=(3.5, 1.0),
    orders=(10, 0),
    epsIncident=1.0,
    epsTransmission=1.45**2,
    sAmplitude=0.0,
    pAmplitude=1.0,
    method="smatrix",
    truncation="circular",
    backend="cuda",
)
```

`LayerStack` stores layers from incident side to transmission side and uses
`z=0` at the top interface, positive downward. When a `z` region cuts through
an existing layer, the builder splits that layer before applying the material
change, so the exported object is still an ordinary RCWA layer list.

`solveStackBatch` and `RCWASimulation` are optional wrappers for special sweep
workflows. They reuse the same `solver.py` CUDA S-matrix core; they are not a
separate physical algorithm.

Field maps expose both amplitudes and squared intensities. Use `EMagnitude`
or `HMagnitude` for `|E|` and `|H|`, `EIntensity` or `HIntensity` for
`|E|^2` and `|H|^2`, and `ENormalizedIntensity` or `HNormalizedIntensity` for
`|E/E0|^2` and `|H/H0|^2`. Paper-style plots that say "self-normalized" should
use `ESelfNormalizedMagnitude` or `HSelfNormalizedMagnitude`; those divide the
returned map by its own maximum and are not the same physical quantity as
incident-normalized intensity.

For sampled scalar layers with curved or oblique material boundaries,
`factorization="auto"` uses a stored normal-vector field when one is available.
The in-plane displacement relation is built as:

```text
D_x, D_y = R(n, t) diag([1 / epsilon]^-1, [epsilon]) R(n, t)^T E_x, E_y
```

where `[1 / epsilon]^-1` is the inverse of the Fourier convolution matrix of
the reciprocal permittivity.  This improves convergence for high-contrast
TM-like fields without entering the tensor anisotropic solver.

For sampled piecewise-constant scalar grids that do not provide a normal field,
the isotropic path now has a lightweight FMMax-inspired fallback: it builds a
normalized contrast map, low-pass filters it to the active Fourier domain,
takes a periodic gradient, and smooths the resulting direction field before the
final unit-vector normalization.  `factorization="auto"` uses this generated
field only for grids that look piecewise constant, while explicit
`factorization="normal-vector"` / `"jones"` requests always try to generate it.
`factorization="standard"` keeps the direct scalar convolution path unchanged.

Power is computed from the real z-directed Poynting flux per Fourier order:

```text
Sz = 0.5 Re(Ex Hy* - Ey Hx*)
```

Reflected power is negated because reflected outgoing waves carry negative
z-directed flux.

## Tensor anisotropic layers

`rcwa3d_anisotropic` supports finite layers with full relative-permittivity
tensors.  The most direct layout for a sampled metasurface layer is:

```text
epsilon.shape == (ny, nx, 3, 3)
```

It also accepts scalar/isotropic inputs, constant `(3, 3)` tensors, and
component mappings such as:

```python
{
    "xx": eps_xx,
    "yy": eps_yy,
    "zz": eps_zz,
    "xz": eps_xz,
    "zx": eps_zx,
}
```

For non-zero xz/zx coupling the solver eliminates `Ez` using the continuous
normal displacement equation:

```text
Dz = Ky Hx - Kx Hy
Ez = [epsilon_zz]^-1 (Dz - [epsilon_zx] Ex - [epsilon_zy] Ey)
```

The inverse is the Li inverse-rule matrix inverse of the Fourier convolution
matrix, not the Fourier transform of `1 / epsilon_zz`.  Substituting this into
`Dx` and `Dy` produces the Schur-complement blocks:

```text
D_x = ([exx] - [exz][ezz]^-1[ezx]) Ex
    + ([exy] - [exz][ezz]^-1[ezy]) Ey
    + [exz][ezz]^-1 Ky Hx - [exz][ezz]^-1 Kx Hy

D_y = ([eyx] - [eyz][ezz]^-1[ezx]) Ex
    + ([eyy] - [eyz][ezz]^-1[ezy]) Ey
    + [eyz][ezz]^-1 Ky Hx - [eyz][ezz]^-1 Kx Hy
```

These blocks feed the full `4N x 4N` first-order matrix for
`[Ex, Ey, Hx, Hy]^T`.  Unlike the isotropic block-off-diagonal eigenproblem,
the tensor system is solved directly because xz/zx coupling introduces
electric-electric and magnetic-magnetic blocks.  Modes are split into forward
and backward subspaces by z-directed Poynting flux for propagating modes and by
the sign of `Im(q)` for evanescent modes.  Layer propagation in the S-matrix
uses separate forward and backward factors:

```text
P_forward  = exp(+i q_forward  k0 d)
P_backward = exp(-i q_backward k0 d)
```

For homogeneous scalar or homogeneous `(3, 3)` tensor layers, the same matrix
formula reduces to independent `4 x 4` eigenproblems for each Fourier order.
The anisotropic solver detects those layers and solves the small per-order
systems directly instead of diagonalizing the full block-diagonal `4N x 4N`
matrix.  This is important for wavelength sweeps with homogeneous dispersive
magneto-optical films and metal mirrors.

The public anisotropic solver now uses the same CUDA-only backend policy as
the isotropic package.  The stable S-matrix path accepts these aliases:

```text
backend=None / "cuda" / "gpu" / "torch" / "torch-cuda" / "auto"
    PyTorch CUDA, required

backend="cpu" / "numpy" / "torch-cpu"
    rejected; there is no silent CPU fallback
```

Public result arrays are converted back to NumPy. CUDA acceleration helps most
when a patterned tensor layer requires a large dense eigenproblem and during
the dense S-matrix interface/cascade solves. Homogeneous tensor helpers still
use small NumPy preprocessing steps before their modal matrices are moved to
CUDA, but the public solve network is assembled and evaluated with CUDA
tensors.

Anisotropic wavelength sweeps intentionally default to one Python worker even
when `workers > 1` is requested. Dense CUDA eigensolves and linear solves
already use optimized native kernels, and running several spectrum points in
Python threads can oversubscribe GPU resources badly. Set
`RCWA3D_ALLOW_THREADED_ANISOTROPIC_SPECTRUM=1` to re-enable threaded wavelength
parallelism.

Process-level wavelength parallelism is rejected for the CUDA-only
anisotropic solver because multiple Python processes contending for one GPU
usually hurts more than it helps; for multi-GPU runs, use one process per GPU
at the application level.

Interfaces are evaluated through the same stable S-matrix solve on the selected
CUDA backend. The CPU-only order-by-order homogeneous helper remains available
internally for regression checks, but it is not part of the public solve path.
By default interface systems are solved as a single multi-RHS LU factorization
plus `lu_solve`, so the left/right excitation blocks share one factorization.
For quick comparisons, set `RCWA3D_INTERFACE_SOLVER=solve` to use direct
`solve` instead of the default `lu_factor + lu_solve` path.

Set `profile=True` on `solveStack` / `solveStackBatch`, or set
`RCWA3D_PROFILE_EIG=1`, to collect per-layer eigensolve diagnostics in
`result.layerEigTimings`. Each entry records the layer index, layer name,
mode path, eig matrix shape, and eigensolve time in seconds.

Homogeneous tensor layers are also evaluated with a batched per-order
eigensolve instead of a Python loop over many tiny `4 x 4` problems. In
wavelength sweeps with dispersive homogeneous anisotropic films, this removes a
large amount of repeated solver overhead without changing the underlying modal
physics.

The public anisotropic S-matrix entrypoint now adds an automatic reduced-space
fast path before assembling the full 2D harmonic system.  If every finite layer
is homogeneous, the stack is solved in the zero-order `4 x 4` subspace and the
resulting amplitudes are embedded back into the requested diffraction-order
array.  If all non-homogeneous layers are one-dimensional, the solver keeps
only the coupled Fourier line (`my=0` for x-varying gratings or `mx=0` for
y-varying gratings) and leaves all off-line diffraction amplitudes at zero.
This is the anisotropic analogue of a `4 x 4` TMM/RCWA automatic route: 1D and
homogeneous problems avoid paying for unused 2D harmonic blocks, while genuine
2D structures such as square rings or cylinders continue through the general
S-matrix RCWA path.

High-level spectra also use the multi-RHS path for all requested incident
states. `polarizations=("TE", "TM")` and custom entries such as
`excitations={"tilted": (0.7 + 0.1j, -0.2j)}` are merged before calling
`solveStackBatch`, so one prepared stack is reused for TE, TM, and custom
incident amplitudes at each wavelength/angle.

For scalar 2D grating layers with curved in-plane material boundaries, callers
may provide `Layer(..., normalField=normal_field)` where `normal_field` has
shape `(ny, nx, 2)`.  In that case the in-plane displacement relation uses a
normal-vector Li factorization:

```text
E_n = n_x E_x + n_y E_y
E_t = -n_y E_x + n_x E_y
D_n = [1 / epsilon]^-1 E_n
D_t = [epsilon] E_t
```

and transforms `D_n, D_t` back to `D_x, D_y`.  The xz/zx magneto-optic
coupling still uses the z-normal Schur-complement path above.  This hybrid is
the practical fast path for structures like a scalar Si cylinder grating on a
homogeneous xz/zx InAs film.

When a sampled scalar grid omits `normalField`, the anisotropic solver now
supports the same Fourier-filtered vector-field generation strategy used by the
isotropic path.  This keeps the public API simple for hand-built scalar grids
while preserving `factorization="standard"` as an explicit escape hatch back to
the plain z-normal tensor Li route.

For circular cylinders, `rcwa3d_anisotropic.AnalyticDisk` avoids sampled
staircasing entirely.  The disk indicator Fourier coefficient is evaluated as:

```text
c_mn = f,                                      G = 0
c_mn = f * 2 J1(|G| r) / (|G| r) * exp(-i G.c), G != 0
f    = pi r^2 / (period_x period_y)
```

`factorization="auto"` is the recommended single path exposed by the geometry
helpers.  It chooses the robust implementation by geometry and material:

- analytic scalar disks use closed-form disk coefficients plus a Jones-vector
  normal/tangent Li transform (`jones-li`), which is the default high-accuracy
  path for circular-cylinder benchmarks;
- sampled scalar shapes with a normal field use normal-vector Li
  (`normal-vector-li`);
- sampled tensor shapes and homogeneous tensors use the general z-normal tensor
  Li path.

`factorization="standard"` keeps the direct closed-form disk coefficients
(`analytic-li`) for fast comparisons.  `factorization="jones"` explicitly
requests the same Jones-vector path used by `auto`.

## Anisotropic Solution Path

The public anisotropic package exposes only `method="smatrix"`. This is the
stable Redheffer scattering-matrix cascade, with homogeneous and 1D reduced
fast paths selected automatically when they are physically equivalent to the
full harmonic system. Requests for `method="etm"`, `method="global"`, or
`method="expm"` are rejected by `solveStack`, `solveStackBatch`,
`RCWASimulation`, and `Project`.

## High-level anisotropic API

Most simulations should use the high-level anisotropic helpers instead of
assembling tensors, sampled grids, compiled layers, and TE/TM loops manually.

Material helpers:

```python
eps = rcwa.gyrotropicXzTensor(
    epsilonParallel=eps_parallel,
    epsilonY=eps_y,
    gyrotropy=g,
    twist=phi,
    twistMode="coupling",  # rotate only xz/zx bias direction
)

eps = rcwa.xzTensor(
    epsilonXx=eps_x,
    epsilonYy=eps_y,
    epsilonZz=eps_z,
    epsilonXz=eps_xz,
    epsilonZx=eps_zx,
    twist=angle,
    twistMode="tensor",  # full R eps R.T tensor rotation
)
```

Geometry helpers:

```python
post = rcwa.circularPostLayer(..., analytic=True)
post = rcwa.rectangularPostLayer(...)
post = rcwa.ellipticalPostLayer(...)
post = rcwa.polygonPostLayer(...)
pyramid = rcwa.slicedTaperStack(kind="rectangle", bottomSize=(w, w), topSize=(0, 0), slices=30)
```

Layer-first patterned construction is also available when you want the code to
read as "create the layer, then draw shapes on that layer":

```python
layer = rcwa.PatternLayer(
    period=(1.0, 1.0),
    thickness=0.08,
    background=1.0,
    shape=(128, 128),
    factorization="auto",
    name="top patterned layer",
)
layer.circle(radius=0.2, material=2.25)
solver_layer = layer.toLayer()
```

The sampled geometry helpers accept scalar or `(3, 3)` tensor materials.  Use
the default `factorization="auto"` unless you are doing a controlled convergence
study.  Scalar sampled shapes with an in-plane normal field enter the NV
Li/FFF path automatically.  Tensor sampled shapes currently use the general
z-normal tensor Li path.

The wrapper `RCWASimulation` precompiles static layers once and keeps
wavelength-dependent homogeneous materials as callables:

```python
simulation = rcwa.RCWASimulation(
    period=(period_x, period_y),
    layers=[
        post,
        rcwa.homogeneousLayer(h_inas, lambda wl: inas_tensor(wl)),
        rcwa.homogeneousLayer(h_metal, metal_epsilon),
    ],
    orders=10,
    truncation="circular",
    backend="auto",
)

result = simulation.solve(wavelength, theta=theta, polarization="TM")
spectra = simulation.spectrum(wavelengths, theta=theta, polarizations=("TE", "TM"))
```
