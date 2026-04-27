# RCWA-3D

一个面向学习和二次开发的三维 RCWA / Fourier Modal Method 实现。

当前支持：

- 各向同性、非磁性材料，允许复介电常数。
- 各向异性介电张量材料，包括常量 `(3, 3)` 张量、采样张量场和常见 xz/zx 磁光耦合张量。
- x/y 双周期、z 方向分层结构，支持 `period_x != period_y` 的矩形周期超表面。
- 任意入射角、s/p 入射振幅。
- 均匀层、二维采样介电常数层，以及几何构建器生成的周期层。
- 反射、透射、衍射级次功率、能量守恒检查。
- 层内 `x-y`、`x-z` 场分布重建。
- 默认使用稳定的 S-matrix / Redheffer 级联；各向同性公共求解路径固定为 CUDA。
- 各向异性路径默认使用一个 `factorization="auto"` 方案：解析圆柱走解析 Fourier/Li，采样标量边界走法向矢量场 NV，张量材料走 z-Li。
- `solveStackBatch` 和 `RCWASimulation` 是可选封装，底层仍复用同一个 `solver.py` S-matrix 核心。

当前限制：

- 尚未实现完整磁性材料、双各向异性 `epsilon/mu/xi/chi` 体系。
- 公共各向异性求解路径只保留稳定的 `method="smatrix"`。
- 2024 eigensystem-free RCWA 的 GPU 矩阵多项式近似和完整 ETM 形式还未作为生产路径接入。
- 高对比二值结构仍需要做收敛性扫描：逐步增加单个 `order`、介电常数采样网格和因子化策略。

## 快速运行

```powershell
python examples\isotropic_example\homogeneousSlab.py
python examples\isotropic_example\binaryGrating.py
python examples\isotropic_example\rectangularMetasurface.py
python examples\isotropic_example\photonicCrystalSlab.py
python examples\isotropic_example\isotropicShapeGallery.py
python examples\isotropic_example\superlensNanocylinderShow.py
python -m unittest discover -s tests
```

各向同性例子都直接在脚本顶部定义运行参数。如果要改 `order`、`grid`、`points` 或 `show`，直接编辑对应脚本里的常量即可。
各向同性求解固定使用 `BACKEND = "cuda"`，不会静默退回 CPU。

## 基本用法

```python
import numpy as np
from rcwa3d_isotropic import Layer, compileLayers, solveStack

I3 = np.eye(3, dtype=complex)
EPS_AIR_TENSOR = 1.0 * I3
EPS_SLAB_TENSOR = 2.25 * I3

layers = compileLayers(
    [Layer(thickness=0.35, epsilon=EPS_SLAB_TENSOR[0, 0])],
    orders=0,
    truncation="circular",
)

result = solveStack(
    layers=layers,
    wavelength=1.0,
    period=(1.0, 1.0),
    orders=0,
    epsIncident=EPS_AIR_TENSOR[0, 0],
    epsTransmission=EPS_AIR_TENSOR[0, 0],
    sAmplitude=1.0,
    pAmplitude=0.0,
    method="smatrix",
    truncation="circular",
    backend="cuda",
)

print(result.reflection, result.transmission, result.conservation)
```

各向同性 `solveStack` 只走 `method="smatrix"` 的 CUDA 路径。

## 各向同性统一建模接口

推荐在新脚本中使用“材料张量 + 完整层栈 + 区域图案化 + `compileLayers` + `solveStack`”这一条路径。材料在脚本里保留为三维介电张量；`LayerStack` 会检查传入的是各向同性 3x3 张量，再写入标量介电常数：

```python
import numpy as np
import rcwa3d_isotropic as rcwa

I3 = np.eye(3, dtype=complex)
EPS_AIR_TENSOR = 1.0 * I3
EPS_SUBSTRATE_TENSOR = 1.45**2 * I3
EPS_SI_TENSOR = 3.4**2 * I3
EPS_AU_TENSOR = (-640.0 + 140.0j) * I3

period = (3.5, 1.0)
siHeight = 0.7
auThickness = 0.1
stripWidth = 0.26

geometry = rcwa.LayerStack(period=period, shape=(8, 512))
geometry.addLayer(siHeight, EPS_AIR_TENSOR, name="air layer patterned with silicon", factorization="standard")
geometry.addLayer(auThickness, EPS_AU_TENSOR, name="gold film", factorization="standard")
geometry.setMaterial(EPS_SI_TENSOR, x=(-stripWidth / 2, stripWidth / 2), y=(-0.5, 0.5), z=(0.0, siHeight))

compiledLayers = rcwa.compileLayers(geometry.toLayers(), orders=(10, 0), truncation="circular")
result = rcwa.solveStack(
    layers=compiledLayers,
    wavelength=4.132,
    period=period,
    orders=(10, 0),
    epsIncident=EPS_AIR_TENSOR[0, 0],
    epsTransmission=EPS_SUBSTRATE_TENSOR[0, 0],
    sAmplitude=0.0,
    pAmplitude=1.0,
    method="smatrix",
    truncation="circular",
    backend="cuda",
)
```

可选项：`solveStackBatch` 可在同一波长/角度下复用一次 prepared stack，同时评估多个入射偏振；`RCWASimulation` 可作为扫谱缓存封装。两者都只是封装，底层算法仍是同一个 `solveStack`/S-matrix 核心。

## 各向异性与快速扫谱

```python
import numpy as np
import rcwa3d_anisotropic as rcwa

post = rcwa.rectangularHollowPostLayer(
    period=(7.7, 7.7),
    thickness=8.13,
    background=1.0,
    post=16.0,
    size=(6.16, 6.16),
    holeRadius=1.54,
    shape=(128, 128),
    factorization="auto",
)

simulation = rcwa.RCWASimulation(
    period=(7.7, 7.7),
    layers=[post],
    orders=5,
    truncation="rectangular",
    backend="auto",
    workers=8,
)

spectra = simulation.spectrum(
    np.linspace(17.3, 18.1, 101),
    theta=np.deg2rad(4.7),
    polarizations=("TE", "TM"),
    bidirectional=True,
)
```

推荐只使用 `factorization="auto"` 这一条路径。它在解析圆柱 benchmark 中避免 Jones/NV 误选，在采样超表面中自动启用 normal-vector Li，在张量材料中使用通用 z-Li 消元。`spectrum` 会在每个波长/入射方向上调用批量求解，因此 TE/TM 会共用一次层模分解和 S-matrix 级联。isotropic 公共求解路径现在固定使用 CUDA；`workers` 只负责分配不同波长的外层任务。

也可以显式调用批量接口：

```python
results = rcwa.solveStackBatch(
    layers=[post],
    wavelength=17.8,
    period=(7.7, 7.7),
    orders=5,
    excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
    truncation="rectangular",
    backend="auto",
)
```

## 几何构建器

单层二维截面仍可直接用 `Pattern2D`：

```python
import numpy as np
from rcwa3d_isotropic import AIR, SI1550, Pattern2D, compileLayers, solveStack

pattern = Pattern2D(period=(0.72, 0.48), shape=(72, 108), background=AIR)
pattern.rectangle(size=(0.34, 0.18), angle=np.deg2rad(25), material=SI1550)
layer = pattern.toLayer(thickness=0.28)

layers = compileLayers([layer], orders=2, truncation="circular")
result = solveStack(
    layers=layers,
    wavelength=1.0,
    period=(0.72, 0.48),
    orders=2,
    backend="cuda",
)
```

三维结构推荐用层栈方式：先铺一层或多层均匀背景，再把体结构写入这些层。`LayerStack` 会按 z 方向自动切片，用多层常截面 RCWA 层近似圆锥面、方锥面或波浪表面。

```python
from rcwa3d_isotropic import LayerStack

geometry = LayerStack(period=(1.0, 1.0), shape=(96, 96))
geometry.addLayer(1.0, AIR, name="air host")

# 圆锥体：z=0 顶部半径小，z=0.8 底部半径大，自动切成 24 层。
geometry.addCone(SI1550, z=(0.0, 0.8), topRadius=0.02, bottomRadius=0.32, slices=24)

# 四面方锥体/方台：矩形截面随 z 线性缩放。
geometry.addPyramid(SI1550, z=(0.1, 0.9), topSize=0.04, bottomSize=(0.45, 0.45), slices=24)

# 波浪形体：由正弦 top surface 限定的三维体。
geometry.addWaveBody(SI1550, baseZ=0.0, meanHeight=0.35, amplitude=0.15, axis="x", slices=24)

# 多边形柱/多边形锥也可直接写入背景层。
geometry.addPolygonPrism(SI1550, z=(0.2, 0.6), vertices=[(-0.2, -0.1), (0.2, -0.1), (0.0, 0.25)])

layers = compileLayers(geometry.toLayers(), orders=3, truncation="circular")
```

各向异性模块也提供同名的 `rcwa3d_anisotropic.LayerStack`，材料可以是标量或 `(3, 3)` 张量。常用三维接口包括：`addVolume`、`addBox`、`addCylinder`、`addCone`、`addPyramid`、`addPolygonPrism`、`addPolygonPyramid`、`addWaveBody`。二维单层形状包括：`circle`、`ellipse`、`rectangle`、`annulus`、`cross`、`stripes`、`polygon`。

## 场分布

场分布计算统一推荐使用 `stackFieldSliceXy`、`stackFieldSliceXz`、`stackFieldComponentsXy` 和 `stackFieldComponentsXz`。这些接口使用全局堆栈坐标：`z=0` 是第一层入射侧界面，有限层位于 `0 <= z <= total_thickness`，`z < 0` 是入射半空间，`z > total_thickness` 是透射半空间。示例脚本里如果要画场，优先直接调用这些接口，不需要自己拼接 Fourier 系数或手动重构 `Ex/Ey/Ez`。

```python
from rcwa3d_isotropic import (
    compileLayers,
    solveStack,
    stackFieldComponentsXz,
    stackFieldSliceXy,
    stackFieldSliceXz,
)

layers = compileLayers([layer], orders=2, truncation="circular")
result = solveStack(
    layers=layers,
    wavelength=1.0,
    period=(0.72, 0.48),
    orders=2,
    returnFields=True,
    truncation="circular",
    backend="cuda",
)

# 过中心 y=0 的整结构 x-z 场图，自动覆盖结构上下半空间。
x, z, e2_xz = stackFieldSliceXz(
    result,
    y=0.0,
    component="EIntensity",
    shape=(241, 181),
)

# 透射侧上方 0.30 um 的 x-y 场图，例如纳米柱顶面上方平面。
total_thickness = sum(layer.thickness for layer in layers)
x, y, e2_xy = stackFieldSliceXy(
    result,
    z=total_thickness + 0.30,
    component="EIntensity",
    shape=(161, 161),
)

# 如果需要完整矢量场，使用 Components 接口一次取回所有分量。
x, z, maps = stackFieldComponentsXz(result, y=0.0, shape=(241, 181))
ex = maps["Ex"]
ey = maps["Ey"]
ez = maps["Ez"]
```

`component` 支持 `Ex/Ey/Ez/Hx/Hy/Hz`，也支持 `EMagnitude/HMagnitude`、`EIntensity/HIntensity`、`ENormalizedIntensity/HNormalizedIntensity`、`ESelfNormalizedMagnitude/HSelfNormalizedMagnitude`。其中 `EIntensity` 使用完整矢量场计算：`|Ex|^2 + |Ey|^2 + |Ez|^2`，不是只取某一个横向分量。

`stackFieldSliceXz` / `stackFieldComponentsXz` 可通过 `xSpan`、`zSpan`、`zPadding`、`y` 和 `shape` 调整范围与采样；`stackFieldSliceXy` / `stackFieldComponentsXy` 可通过 `xSpan`、`ySpan`、`z` 和 `shape` 调整平面位置与采样。若只想看某个有限层内部的局部坐标场，也可以使用旧接口 `fieldSliceXy`、`fieldSliceXz`、`fieldComponentsXy`、`fieldComponentsXz`，其中 `z` 是所选层内部的局部坐标。

论文里写 self-normalized 的场图应使用 `ESelfNormalizedMagnitude` 或 `HSelfNormalizedMagnitude`，不要和 `|E/E0|^2`、`|H/H0|^2` 混用。

显示工具在 `rcwa3d_isotropic.visualization` 和 `rcwa3d_isotropic.display` 中：`plotEpsilon`、`plotSpectrum`、`plotField`。
