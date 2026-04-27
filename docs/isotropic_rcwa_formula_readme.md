# Isotropic RCWA README (CUDA + Formula)

本文件面向 `rcwa3d_isotropic`，回答两个问题：

1. 是否使用 CUDA 加速；
2. 是否按主流论文中的 RCWA/FMM 步骤实现，并给出对应公式。

---

## 1) 结论

- `是`，各向同性求解路径使用 CUDA（PyTorch CUDA）加速。
- `是`，算法主干按经典 RCWA/FMM + S-matrix（Redheffer 级联）路线实现，并结合了工程增强（如自动降维、解析几何卷积、normal-vector/Jones 因子分解）。

这表示：核心物理步骤与主流论文一致，但并非“逐字复现某一篇单独论文代码”，而是一个工程化整合实现。

---

## 2) CUDA 路径对应代码

- 后端解析：`src/rcwa3d_isotropic/backend.py`
  - `resolveBackend(...)` 将 `None/auto/torch/gpu/cuda` 映射到 CUDA。
  - CPU 选项会被拒绝（不静默回退）。
- 主入口：`src/rcwa3d_isotropic/solver.py`
  - `solveStack(...)` 默认 `backend="cuda"`，并调用 `prepareStackTorch(...)`、`evaluatePreparedStackTorch(...)`。
  - `_requireSMatrixMethod(...)` 限定当前各向同性公共路径为 `method="smatrix"`。

---

## 3) 算法流程与公式（按实现顺序）

下面的符号和流程与 `solver.py / smatrix.py / fields.py` 对应。

### 3.1 傅里叶谐波与横向波矢

入射侧折射率 \(n_{inc}=\sqrt{\varepsilon_{inc}}\)，自由空间波数 \(k_0=2\pi/\lambda\)。

\[
K_x = n_{inc}\sin\theta\cos\phi + m_x\frac{\lambda}{p_x},\quad
K_y = n_{inc}\sin\theta\sin\phi + m_y\frac{\lambda}{p_y}
\]

谐波截断支持 rectangular/circular（工程上 circular 常有更高效率）。

### 3.2 状态变量与一阶矩阵方程

切向场状态向量：

\[
\mathbf{f} = [E_x, E_y, H_x, H_y]^T
\]

层内一阶系统：

\[
\frac{d\mathbf{f}}{d(k_0 z)} = i\mathbf{A}\mathbf{f}
\]

在各向同性标量介质卷积矩阵 \(\mathbf{E}\) 下，按代码实现可写为：

\[
\begin{bmatrix}
\mathbf{E}_x'\\
\mathbf{E}_y'
\end{bmatrix}
=
\mathbf{P}
\begin{bmatrix}
\mathbf{H}_x\\
\mathbf{H}_y
\end{bmatrix},\quad
\begin{bmatrix}
\mathbf{H}_x'\\
\mathbf{H}_y'
\end{bmatrix}
=
\mathbf{Q}
\begin{bmatrix}
\mathbf{E}_x\\
\mathbf{E}_y
\end{bmatrix}
\]

其中（与 `pqMatrices(...)` 对应）：

\[
\mathbf{P}_{11}=K_x\mathbf{E}^{-1}K_y,\quad
\mathbf{P}_{12}=\mathbf{I}-K_x\mathbf{E}^{-1}K_x
\]
\[
\mathbf{P}_{21}=K_y\mathbf{E}^{-1}K_y-\mathbf{I},\quad
\mathbf{P}_{22}=-K_y\mathbf{E}^{-1}K_x
\]
\[
\mathbf{Q}_{11}=-K_xK_y,\quad
\mathbf{Q}_{12}=K_xK_x-\mathbf{E}
\]
\[
\mathbf{Q}_{21}=\mathbf{E}-K_yK_y,\quad
\mathbf{Q}_{22}=K_yK_x
\]

### 3.3 模态特征分解（每层）

代码中通过 \(\mathbf{P}\mathbf{Q}\) 求特征值，得到纵向传播常数 \(q\) 与模态矩阵。

\[
\mathbf{P}\mathbf{Q}\mathbf{W}=\mathbf{W}\mathbf{\Lambda},\quad
q=\sqrt{\Lambda}
\]

并按前向分支选取 \(q\)（`forwardKz(...)` 分支规则）。

### 3.4 接口散射矩阵

在每个界面满足切向场连续。令左右区域前/后向模态基分别为
\(\mathbf{F}^{+},\mathbf{F}^{-}\)，
通过线性方程解出接口 S 矩阵（`interfaceSMatrix(...)`）：

\[
\begin{bmatrix}
\mathbf{b}_L\\
\mathbf{a}_R
\end{bmatrix}

=
\begin{bmatrix}
\mathbf{S}_{11} & \mathbf{S}_{12}\\
\mathbf{S}_{21} & \mathbf{S}_{22}
\end{bmatrix}
\begin{bmatrix}
\mathbf{a}_L\\
\mathbf{b}_R
\end{bmatrix}
\]

### 3.5 层内传播矩阵

厚度 \(d\) 的层传播（`propagationSMatrix(...)`）：

\[
\mathbf{P}_{prop}=\mathrm{diag}\left(e^{i q_n k_0 d}\right)
\]

### 3.6 Redheffer 星积级联

总结构 S 矩阵由界面与传播矩阵通过 Redheffer 星积级联（`redhefferStar(...)`、`cascadeMany(...)`）：

\[
\mathbf{S}_{tot}=\mathbf{S}_1 \star \mathbf{S}_2 \star \cdots \star \mathbf{S}_N
\]

这是 RCWA 稳定实现的关键步骤，避免了直接传输矩阵在倏逝模下的数值爆炸。

### 3.7 反射/透射与衍射级次功率

给定入射激励 \((s,p)\) 后：

\[
\mathbf{r}=\mathbf{S}_{11}\mathbf{a}_{inc},\quad
\mathbf{t}=\mathbf{S}_{21}\mathbf{a}_{inc}
\]

用实部 Poynting 通量计算功率（`flux(...)`）：

\[
S_z=\frac{1}{2}\Re\left(E_x H_y^* - E_y H_x^*\right)
\]

得到总 \(R,T\) 与各衍射级功率（`orderResults(...)`）。

### 3.8 场分布重建

场重建统一走 `fields.py`：

- `stackFieldSliceXz / stackFieldComponentsXz`
- `stackFieldSliceXy / stackFieldComponentsXy`

傅里叶重构（`reconstructFourierGrid(...)`）：

\[
F(x,y)=\sum_n \hat{F}_n \exp\left[i k_0 (k_{x,n}x + k_{y,n}y)\right]
\]

电场强度：

\[
|E|^2 = |E_x|^2 + |E_y|^2 + |E_z|^2
\]

---

## 4) 与“当前论文算法”的关系

各向同性路径严格落在主流 RCWA 文献框架内：

- Fourier Modal Method / RCWA；
- Li 因子分解家族（含 normal-vector/Jones 工程化路径）；
- S-matrix + Redheffer 稳定级联；
- 基于 Poynting 通量的能量守恒与级次功率计算。

工程增强部分（自动降维、解析几何卷积、批量激励复用、统一场重建接口）是为收敛性和可用性服务，不改变上述物理主干。

---

## 5) 最小调用模板（CUDA）

```python
import rcwa3d_isotropic as rcwa

compiled = rcwa.compileLayers(layers, orders=(Nx, Ny), truncation="circular")
result = rcwa.solveStack(
    layers=compiled,
    wavelength=wl,
    period=(px, py),
    orders=(Nx, Ny),
    epsIncident=eps_in,
    epsTransmission=eps_out,
    sAmplitude=0.0,
    pAmplitude=1.0,
    returnFields=True,
    method="smatrix",
    backend="cuda",
)

# 整结构 x-z 场图
x, z, e2 = rcwa.stackFieldSliceXz(result, component="EIntensity", shape=(241, 181))
```

