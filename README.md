# RCWA-3D

RCWA-3D 是一个面向学习、验证和二次开发的三维 RCWA / Fourier Modal Method 实现。项目把各向同性标量介质和各向异性张量介质分成两个包：

- `rcwa3d_isotropic`：各向同性、非磁性、标量介电常数 RCWA。公共求解路径固定为 CUDA + PyTorch + S-matrix。
- `rcwa3d_anisotropic`：各向异性介电张量 RCWA，支持常量张量、采样张量场、常见 xz/zx 磁光耦合张量，同样使用 CUDA S-matrix 作为生产路径。

项目当前重点是数值稳定性、可读性和可扩展性。默认不提供静默 CPU fallback：如果 CUDA 不可用，求解器会直接报错，避免同一脚本在不同机器上走到不同数值路径。

## 功能概览

当前支持：

- x/y 双周期、z 方向分层结构。
- `period_x != period_y` 的矩形周期超表面。
- rectangular 和 circular Fourier 谐波截断。
- 任意入射角 `theta/phi` 与 s/p 入射振幅。
- TE/TM 和自定义 s/p 激励批量求解。
- 均匀层、二维采样标量层、解析几何层、三维分片几何层。
- 各向同性解析圆、椭圆、矩形、环形等 Fourier 卷积。
- 采样标量边界的 normal-vector Li / Jones 类因子化。
- 各向异性常量 `(3, 3)` 张量、采样 `(ny, nx, 3, 3)` 张量场、xz/zx 磁光耦合张量。
- 稳定 S-matrix / Redheffer 星积级联。
- 自动齐次层快速路径和一维结构降维路径。
- 反射、透射、衍射级次功率和能量守恒检查。
- 全堆栈 `x-y`、`x-z` 场分布重建。
- 谱扫描封装 `RCWASimulation.spectrum(...)`。
- 层模和堆栈准备时间的 profile 信息。

当前限制：

- 暂未实现完整磁性材料与双各向异性 `epsilon/mu/xi/chi` 体系。
- 公共求解路径只保留稳定的 `method="smatrix"`；`etm/global/expm` 等旧方法不会作为生产入口暴露。
- 高对比二值结构仍需要做收敛性扫描：逐步增加 `orders`、采样网格、分片数和因子化策略。
- 各向同性公共 API 已经转向 `RCWASimulation`；旧式 `rcwa3d_isotropic.compileLayers/solveStack` 不再作为公共接口使用。

## 安装与 CUDA

推荐环境：

- Python 3.9 或更高版本。
- NVIDIA GPU。
- 可用的 NVIDIA 驱动。
- CUDA 版 PyTorch。

创建并进入虚拟环境：

```powershell
cd D:\RCWA-python-codex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 拒绝执行激活脚本，可在当前用户范围内放宽策略：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

安装 CUDA 版 PyTorch。请优先使用 PyTorch 官方安装选择器生成的命令：

<https://docs.pytorch.org/get-started/locally/>

例如官方选择器给出 CUDA 12.8 wheel 时，可使用：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

然后安装本项目：

```powershell
python -m pip install -e ".[gpu]"
```

验证 CUDA：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import rcwa3d_isotropic as rcwa; b = rcwa.resolveBackend('cuda'); print(b.name, b.isCuda, b.device)"
```

期望看到：

```text
cuda available: True
cuda True cuda
```

`nvidia-smi` 中的 `CUDA Version` 表示驱动最高支持的 CUDA runtime 能力，不等于已经安装了同版本 CUDA Toolkit。使用 PyTorch 官方预编译 wheel 时通常不需要单独安装 CUDA Toolkit。

## 快速运行

运行测试：

```powershell
python -m unittest discover -s tests
```

运行各向同性示例：

```powershell
python examples\isotropic_example\homogeneousSlab.py
python examples\isotropic_example\binaryGrating.py
python examples\isotropic_example\rectangularMetasurface.py
python examples\isotropic_example\photonicCrystalSlab.py
python examples\isotropic_example\isotropicShapeGallery.py
python examples\isotropic_example\superlensNanocylinderShow.py
python examples\isotropic_example\lawp2020BinaryGratingSpectrum.py
python examples\isotropic_example\applsci2019AuSiMetagratingField.py
```

各向同性 example 都在脚本顶部定义参数，例如：

```python
BACKEND = "cuda"
PRECOMPILE = True
CACHE_MODES = True
WORKERS = 1
ORDER = 4
POINTS = 501
```

谱扫描类 example 会显式调用：

```python
spectrum = simulation.spectrum(wavelengths, polarizations=("TE", "TM"), workers=WORKERS)
```

在当前各向同性实现中，`WORKERS = 1` 是默认推荐值：满足静态层、谐波数适中、无自动降维等条件时，`spectrum(...)` 会走 GPU batched powers 路径。`workers > 1` 可用于逐波长外层并行，但通常不如单 GPU 批量路径稳定可控。

## 各向同性基本用法

推荐新代码直接使用 `RCWASimulation`。这是当前各向同性公共建模和求解入口。

```python
import rcwa3d_isotropic as rcwa

layer = rcwa.Layer(
    thickness=0.30,
    epsilon=2.25,
    name="glass slab",
)

simulation = rcwa.RCWASimulation(
    period=(1.0, 1.0),
    layers=[layer],
    orders=3,
    truncation="circular",
    epsIncident=1.0,
    epsTransmission=1.0,
    method="smatrix",
    backend="cuda",
    precompile=True,
    cacheModes=True,
)

result = simulation.solve(1.0, polarization="TE")

print(result.reflection)
print(result.transmission)
print(result.conservation)
```

多个激励共享一次 prepared stack：

```python
results = simulation.solveExcitations(
    1.0,
    {
        "TE": (1.0, 0.0),
        "TM": (0.0, 1.0),
        "custom": (0.7 + 0.1j, -0.2j),
    },
)
```

谱扫描：

```python
import numpy as np

wavelengths = np.linspace(0.8, 1.2, 201)
spectrum = simulation.spectrum(
    wavelengths,
    theta=0.0,
    phi=0.0,
    polarizations=("TE", "TM"),
    workers=1,
)

reflection_te = spectrum["TE"]["reflection"]
transmission_te = spectrum["TE"]["transmission"]
conservation_te = spectrum["TE"]["conservation"]
```

profile 单点计算：

```python
profile = simulation.solve(1.0, polarization="TM", profile=True)

for timing in profile.layerEigTimings:
    print(
        timing.layerIndex,
        timing.kind,
        timing.matrixShape,
        timing.factorizationTimeSeconds,
        timing.inverseTimeSeconds,
        timing.pqTimeSeconds,
        timing.eigTimeSeconds,
    )

print(profile.stackTiming)
```

## 各向同性几何

解析几何优先用于标准形状，可减少采样 staircasing 误差：

```python
import rcwa3d_isotropic as rcwa

layer = rcwa.circularPostLayer(
    period=(0.8, 0.8),
    thickness=0.25,
    background=1.0,
    post=3.4**2,
    radius=0.18,
    analytic=True,
    factorization="auto",
    name="analytic circular post",
)
```

`Pattern2D` 适合手动构建二维采样单层：

```python
import numpy as np
import rcwa3d_isotropic as rcwa

pattern = rcwa.Pattern2D(
    period=(0.72, 0.48),
    shape=(72, 108),
    background=rcwa.AIR.epsilon(),
)
pattern.rectangle(
    size=(0.34, 0.18),
    angle=np.deg2rad(25),
    material=rcwa.SI1550.epsilon(),
)
layer = pattern.toLayer(thickness=0.28)
```

三维结构推荐使用 `LayerStack`：先铺背景层，再把体结构写入指定 `x/y/z` 区域。`LayerStack` 会在 z 方向自动切层，最终仍输出普通 RCWA 层列表。

```python
import rcwa3d_isotropic as rcwa

geometry = rcwa.LayerStack(period=(1.0, 1.0), shape=(96, 96))
geometry.addLayer(1.0, rcwa.AIR.epsilon(), name="air host")
geometry.addCone(
    rcwa.SI1550.epsilon(),
    z=(0.0, 0.8),
    topRadius=0.02,
    bottomRadius=0.32,
    slices=24,
)
geometry.addPyramid(
    rcwa.SI1550.epsilon(),
    z=(0.1, 0.9),
    topSize=0.04,
    bottomSize=(0.45, 0.45),
    slices=24,
)
geometry.addPolygonPrism(
    rcwa.SI1550.epsilon(),
    z=(0.2, 0.6),
    vertices=[(-0.2, -0.1), (0.2, -0.1), (0.0, 0.25)],
)

simulation = rcwa.RCWASimulation(
    period=geometry.period,
    layers=geometry.toLayers(),
    orders=3,
    truncation="circular",
    backend="cuda",
)
```

常用几何 helper：

- `homogeneousLayer`
- `circularPostLayer`
- `ellipticalPostLayer`
- `rectangularPostLayer`
- `rectangularHollowPostLayer`
- `annularPostLayer`
- `crossPostLayer`
- `polygonPostLayer`
- `photonicCrystalSlab`
- `LayerStack`
- `PatternLayer`

## 各向同性算法说明

设自由空间波数为：

```text
k0 = 2 pi / wavelength
```

入射介质折射率：

```text
n_inc = sqrt(epsIncident)
```

每个 Fourier 级次的归一化横向波矢：

```text
Kx = n_inc sin(theta) cos(phi) + mx wavelength / period_x
Ky = n_inc sin(theta) sin(phi) + my wavelength / period_y
```

rectangular truncation 保留：

```text
mx = -Nx ... Nx
my = -Ny ... Ny
```

circular truncation 保留缩放倒空间圆盘内的整数级次：

```text
(mx / Nx)^2 + (my / Ny)^2 <= 1
```

有限层内使用切向状态向量：

```text
f = [Ex, Ey, Hx, Hy]^T
```

一阶系统：

```text
d f / d(k0 z) = i A f
```

对各向同性标量介电卷积矩阵 `E`，代码把方程写成块形式：

```text
[Ex']   [ P11 P12 ] [Hx]
[Ey'] = [ P21 P22 ] [Hy]

[Hx']   [ Q11 Q12 ] [Ex]
[Hy'] = [ Q21 Q22 ] [Ey]
```

其中：

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

层模来自：

```text
P Q W = W Lambda
q = sqrt(Lambda)
```

分支选择使用前向传播约定。传播因子为：

```text
exp(i q k0 d)
```

堆栈级联不直接使用传输矩阵，而是使用稳定的 interface S-matrix、propagation S-matrix 和 Redheffer 星积：

```text
S_total = S_1 star S_2 star ... star S_N
```

给定入射列向量 `a_inc`：

```text
r = S11 a_inc
t = S21 a_inc
```

功率由 z 向 Poynting 通量计算：

```text
Sz = 0.5 Re(Ex Hy* - Ey Hx*)
```

反射波的 z 向通量为负，因此总反射功率会取负号归一化。

### 各向同性优化路径

当前各向同性底层有几条重要优化：

- 齐次标量层跳过密集 RCWA 特征分解，直接构造每个级次的 s/p 均匀介质基。
- 一维结构自动降维，只保留真正耦合的 Fourier 线，并把结果嵌回完整级次数组。
- 谱扫描在条件允许时走 batched GPU eigensolve 和 powers-only 路径，只返回反射/透射/守恒，不构造场系数。
- `prepareStackPowersTorch` 只准备功率所需的反射矩阵和前向算子，避免完整 S-matrix 级联里用不到的列。
- P/Q 矩阵组装使用预分配 block 写入，减少 `torch.cat` 和显式对角矩阵构造。
- `RCWASimulation` 对静态层做 CUDA 预编译，并用 LRU 缓存 prepared stack。

### Li 因子化与 normal-vector 路径

对于采样标量层，`factorization="standard"` 使用直接标量卷积。对于高对比边界，推荐使用 `factorization="auto"`。

当几何提供法向场，或采样网格看起来是 piecewise constant 时，求解器会使用 normal-vector Li 因子化。局部法向/切向分解为：

```text
E_n = n_x E_x + n_y E_y
E_t = -n_y E_x + n_x E_y
D_n = [1 / epsilon]^-1 E_n
D_t = [epsilon] E_t
```

再把 `D_n, D_t` 变回 `D_x, D_y`。这对高对比 TM-like 场收敛更友好。

若采样标量网格没有显式 `normalField`，`auto` 会为 piecewise constant 网格生成一个 Fourier-filtered 法向场：先从材料 contrast map 得到周期梯度，再平滑和归一化方向。`standard` 会显式关闭这个生成路径。

## 各向异性基本用法

?????? API ??? PyTorch/CUDA `RCWASimulation`??? `solveStack/solveStackBatch/compileLayers` ????????????? `RCWASimulation(precompile=True, cacheModes=True)` ?????

```python
import numpy as np
import rcwa3d_anisotropic as rcwa

layer = rcwa.Layer(
    thickness=0.018,
    epsilon=rcwa.xzTensor(2.2, 2.4, 2.1, 0.04, 0.04),
    name="thin xz tensor film",
)

simulation = rcwa.RCWASimulation(
    period=(0.9, 1.1),
    layers=[layer],
    orders=(1, 1),
    epsIncident=1.0,
    epsTransmission=1.0,
    truncation="circular",
    backend="cuda",
    precompile=True,
    cacheModes=True,
)
result = simulation.solveExcitation(
    1.05,
    theta=np.deg2rad(3.0),
    phi=np.deg2rad(11.0),
    sAmplitude=0.0,
    pAmplitude=1.0,
)

print(result.reflection, result.transmission, result.conservation, result.solvedBy)
```

高层扫谱：

```python
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
    backend="cuda",
)

spectra = simulation.spectrum(
    np.linspace(17.3, 18.1, 101),
    theta=np.deg2rad(4.7),
    polarizations=("TE", "TM"),
    bidirectional=True,
)
```

各向异性材料 helper：

```python
eps = rcwa.xzTensor(
    epsilonXx=2.2,
    epsilonYy=2.4,
    epsilonZz=2.1,
    epsilonXz=0.04,
    epsilonZx=0.04,
)

eps = rcwa.gyrotropicXzTensor(
    epsilonParallel=2.2,
    epsilonY=2.4,
    gyrotropy=0.03j,
    twist=np.deg2rad(30),
    twistMode="coupling",
)
```

各向异性几何 helper 支持标量材料和 `(3, 3)` 张量材料：

- `circularPostLayer`
- `rectangularPostLayer`
- `rectangularHollowPostLayer`
- `ellipticalPostLayer`
- `polygonPostLayer`
- `slicedTaperStack`
- `PatternLayer`
- `LayerStack`

`factorization="auto"` 是推荐路径：

- 采样标量形状有 normal field 时使用 normal-vector Li。
- 采样张量形状和齐次张量层使用 z-normal tensor Li。
- `factorization="standard"` 可用于对照，保留直接采样卷积路径。

## 各向异性算法说明

各向异性层可接受：

```text
epsilon.shape == (3, 3)
epsilon.shape == (ny, nx, 3, 3)
```

也可接受 component mapping，例如：

```python
{
    "xx": eps_xx,
    "yy": eps_yy,
    "zz": eps_zz,
    "xz": eps_xz,
    "zx": eps_zx,
}
```

对于 xz/zx 耦合，求解器先用连续法向位移消去 `Ez`：

```text
Dz = Ky Hx - Kx Hy
Ez = [epsilon_zz]^-1 (Dz - [epsilon_zx] Ex - [epsilon_zy] Ey)
```

这里的 `[epsilon_zz]^-1` 是 Fourier 卷积矩阵逆，不是 `1 / epsilon_zz` 的 Fourier 变换。

代入 `Dx/Dy` 后得到 Schur complement 块：

```text
D_x = ([exx] - [exz][ezz]^-1[ezx]) Ex
    + ([exy] - [exz][ezz]^-1[ezy]) Ey
    + [exz][ezz]^-1 Ky Hx - [exz][ezz]^-1 Kx Hy

D_y = ([eyx] - [eyz][ezz]^-1[ezx]) Ex
    + ([eyy] - [eyz][ezz]^-1[ezy]) Ey
    + [eyz][ezz]^-1 Ky Hx - [eyz][ezz]^-1 Kx Hy
```

这些块组成完整 `4N x 4N` 一阶矩阵。由于 xz/zx 耦合会产生 electric-electric 和 magnetic-magnetic blocks，各向异性系统不能像标量各向同性那样只求 `P Q`，而是直接求完整一阶系统。

模态按 z 向 Poynting flux 和 evanescent 衰减方向分成 forward/backward 子空间。传播因子：

```text
P_forward  = exp(+i q_forward  k0 d)
P_backward = exp(-i q_backward k0 d)
```

齐次张量层会降为每个 Fourier 级次的 `4 x 4` 小特征问题，并使用批量 per-order eigensolve，避免构造大型 block diagonal 问题。

各向异性也使用 S-matrix / Redheffer 级联，并有自动 reduced-space 快速路径：

- 全部有限层齐次时，只解零级 `4 x 4` 子空间，再嵌回完整级次数组。
- 一维结构只保留耦合 Fourier 线。
- 真正二维结构继续走通用 S-matrix RCWA。

## 场分布重建

推荐使用全堆栈坐标接口：

- `stackFieldSliceXy`
- `stackFieldSliceXz`
- `stackFieldComponentsXy`
- `stackFieldComponentsXz`

`z=0` 是第一层入射侧界面。有限层位于：

```text
0 <= z <= total_thickness
```

`z < 0` 是入射半空间，`z > total_thickness` 是透射半空间。

```python
x, z, e2 = rcwa.stackFieldSliceXz(
    result,
    y=0.0,
    component="EIntensity",
    shape=(241, 181),
)

x, z, maps = rcwa.stackFieldComponentsXz(
    result,
    y=0.0,
    shape=(241, 181),
)

ex = maps["Ex"]
ey = maps["Ey"]
ez = maps["Ez"]
```

Fourier 重建形式：

```text
F(x, y) = sum_n F_n exp(i k0 (Kx_n x + Ky_n y))
```

纵向场由 Maxwell 约束恢复。例如各向同性层中：

```text
Hz = Kx Ey - Ky Ex
Ez = E^-1 (Ky Hx - Kx Hy)
```

`component` 支持：

- `Ex/Ey/Ez/Hx/Hy/Hz`
- `EMagnitude/HMagnitude`
- `EIntensity/HIntensity`
- `ENormalizedIntensity/HNormalizedIntensity`
- `ESelfNormalizedMagnitude/HSelfNormalizedMagnitude`

其中：

```text
EIntensity = |Ex|^2 + |Ey|^2 + |Ez|^2
HIntensity = |Hx|^2 + |Hy|^2 + |Hz|^2
```

论文中标注 self-normalized 的图应使用 `ESelfNormalizedMagnitude` 或 `HSelfNormalizedMagnitude`。它们是“图内最大值归一化”的显示量，不等同于 `|E/E0|^2` 或 `|H/H0|^2`。

显示工具：

- `rcwa3d_isotropic.visualization.plotEpsilon`
- `rcwa3d_isotropic.visualization.plotSpectrum`
- `rcwa3d_isotropic.visualization.plotField`

## 后端策略

各向同性和各向异性公共求解路径都使用 CUDA-only 策略。

接受：

```text
backend=None
backend="auto"
backend="cuda"
backend="gpu"
backend="torch"
backend="torch-cuda"
```

这些都会解析到 PyTorch CUDA。

拒绝：

```text
backend="cpu"
backend="numpy"
backend="torch-cpu"
```

拒绝 CPU 后端是有意设计：测试、示例和生产脚本应该暴露环境问题，而不是静默切换到另一条数值路径。

## 常见问题

### `RuntimeError: requires a CUDA-enabled torch`

通常原因：

- 当前环境安装的是 CPU 版 PyTorch。
- IDE 使用了另一个 Python 解释器。
- NVIDIA 驱动不可用或版本过旧。
- 没有可见 NVIDIA GPU。

检查：

```powershell
where python
python -m pip show torch
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### `ValueError: CUDA-only`

这是预期行为。把脚本里的后端保持为：

```python
BACKEND = "cuda"
```

不要把 public example 改成 `"cpu"`、`"numpy"` 或 `"torch-cpu"`。

### CUDA out of memory

RCWA 显存压力主要来自：

- Fourier 阶数。
- 二维谐波数量。
- 层数。
- 谱扫描点数。
- 场重建采样网格。

处理方式：

- 降低 `ORDER` 或把 `(Nx, Ny)` 改成一维阶数。
- 使用 circular truncation。
- 减少采样网格或谱点数。
- 先关闭 `returnFields`。
- 对一维问题确认是否触发自动降维路径。
- 关闭其他占用 GPU 显存的程序。

### 谱扫描慢

优先检查：

- 各向同性 spectrum example 是否使用 `WORKERS = 1`。
- 层是否是静态层。波长相关 layer factory 会让 batched spectrum 回退到逐点路径。
- 谐波数量是否过大。当前 batched powers 路径只对适中规模自动启用。
- 是否请求了场分布。谱扫描通常只需要 powers，不要在 sweep 内做 field solve。

### 能量守恒误差偏大

常见原因：

- Fourier 阶数不足。
- 高对比边界使用了 `factorization="standard"`。
- 采样网格太粗。
- 结构几何不够光滑或分片太少。
- 材料有吸收但仍按无吸收期望检查 `R + T = 1`。

建议逐步扫描 `orders`、采样网格和 factorization，并观察 `result.conservation`。

## 测试与维护

运行全部测试：

```powershell
python -m unittest discover -s tests
```

运行示例风格检查包含在测试集中。它会确保各向同性 example：

- 使用 `BACKEND = "cuda"`。
- 定义 `PRECOMPILE` 和 `CACHE_MODES`。
- 使用 `RCWASimulation`。
- 谱扫描脚本定义 `WORKERS`，并在构造器和 `spectrum(...)` 中传入 `workers=WORKERS`。
- 不重新引入旧式 `solveStack/solveStackBatch/compileLayers`、CPU backend 或命令行 argparse 入口。

维护建议：

- 新的各向同性脚本优先使用 `RCWASimulation`。
- 新的谱扫描示例优先使用 `WORKERS = 1`。
- 需要对比旧算法时写内部测试，不要把旧入口重新暴露到 public API。
- 添加新几何 helper 时同步添加 field 或 solver path 测试。
- 修改 CUDA/PyTorch 后先运行 CUDA 验证命令，再运行完整测试。

## 项目结构

```text
src/
  rcwa3d_isotropic/
    analytic.py        解析标量几何 Fourier 系数
    backend.py         CUDA 后端解析
    builder.py         PatternLayer / LayerStack
    fields.py          场重建
    fourier.py         谐波枚举、Kx/Ky、卷积矩阵
    geometry.py        各向同性几何 helper
    materials.py       常用标量材料
    simulation.py      当前推荐的高层各向同性接口
    solver.py          各向同性 CUDA S-matrix 核心
    factorization.py   标量 Li / normal-vector 因子化

  rcwa3d_anisotropic/
    backend.py         CUDA 后端解析
    builder.py         张量几何构建
    factorization.py   张量 Li 因子化
    geometry.py        各向异性几何 helper
    materials.py       张量材料 helper
    project.py         高层 Project/AnisotropicRCWA 封装
    simulation.py      高层各向异性扫谱接口
    solver.py          各向异性 CUDA S-matrix 核心

examples/
  isotropic_example/   各向同性示例
  anisotropic_example/ 各向异性示例

tests/                 单元测试和路径回归测试
```

本 README 是项目唯一维护文档。算法说明、CUDA 配置、示例规范和维护约定都集中在这里。
