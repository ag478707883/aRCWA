# Anisotropic Stable S-Matrix Path

The public anisotropic RCWA API now routes directly through the S-matrix
implementation in `src/rcwa3d_anisotropic/solver.py`.  The old
`rcwa3d_anisotropic.methods` dispatcher has been removed so there is a single
production solve path to maintain.

Use `method="smatrix"` or omit `method`; other public method choices are
rejected so production runs cannot silently move onto a less stable numerical
path.

```python
import numpy as np
import rcwa3d_anisotropic as rcwa

layer = rcwa.Layer(
    thickness=0.018,
    epsilon=rcwa.xzTensor(2.2, 2.4, 2.1, 0.04, 0.04),
    name="thin xz tensor film",
)

result = rcwa.solveStack(
    layers=[layer],
    wavelength=1.05,
    period=(0.9, 1.1),
    orders=(1, 1),
    epsIncident=1.0,
    epsTransmission=1.0,
    theta=np.deg2rad(3.0),
    phi=np.deg2rad(11.0),
    sAmplitude=0.0,
    pAmplitude=1.0,
    truncation="circular",
    backend="cuda",
)

print(result.reflection, result.transmission, result.conservation, result.solvedBy)
```

`backend=None`, `"cuda"`, `"gpu"`, `"torch"`, `"torch-cuda"`, and `"auto"`
all resolve to PyTorch CUDA. `"cpu"`, `"numpy"`, and `"torch-cpu"` are rejected;
there is no CPU fallback in the public anisotropic solve path.

## Validation

The tests check that:

- public single-excitation solves route to the stable S-matrix path;
- unsupported public methods raise a clear error;
- batch results match single-excitation results;
- prepared S-matrix components stay on CUDA tensors during the solve.

Run:

```powershell
python -m unittest discover -s tests
```
