# CUDA 配置文档

本文档用于配置 RCWA-3D 项目的 CUDA 运行环境，服务各向同性和各向异性 RCWA 的 CUDA-only 求解路径。

## 1. 项目中的 GPU 后端

本项目有两套后端逻辑，需要先区分清楚：

| 模块 | GPU 依赖 | 默认策略 | 说明 |
| --- | --- | --- | --- |
| `rcwa3d_isotropic` | PyTorch CUDA | CUDA-only | 各向同性求解器不再静默回退 CPU。`backend=None`、`"cuda"`、`"gpu"`、`"torch"`、`"torch-cuda"`、`"auto"` 都会解析到 CUDA PyTorch。 |
| `rcwa3d_anisotropic` | PyTorch CUDA | CUDA-only | 各向异性求解器也统一到稳定 S-matrix CUDA 路径；`backend=None`、`"cuda"`、`"gpu"`、`"torch"`、`"torch-cuda"`、`"auto"` 都会解析到 CUDA PyTorch。 |

因此，如果运行示例，例如：

```powershell
python examples\isotropic_example\photonicCrystalSlab.py
python examples\isotropic_example\rectangularMetasurface.py
python examples\isotropic_example\applsci2019AuSiMetagratingField.py
```

必须保证当前 Python 环境里安装的是可用的 CUDA 版 PyTorch，并且机器上有可见的 NVIDIA GPU。

## 2. 前置条件

推荐环境：

- NVIDIA GPU。
- 已安装 NVIDIA 显卡驱动，且驱动版本满足所选 PyTorch CUDA runtime 的最低要求。
- Python 3.9 或更高版本。
- `pip` 可用。
- 项目环境使用同一个 Python 解释器运行安装、测试和 IDE。

注意：

- `nvidia-smi` 里显示的 `CUDA Version` 表示当前驱动最高支持的 CUDA 能力，不等于你已经安装了同版本 CUDA Toolkit，也不等于 PyTorch 正在使用的 CUDA runtime。
- 使用 PyTorch 官方预编译 wheel 时，通常不需要单独安装 CUDA Toolkit；只有编译自定义 CUDA/C++ 扩展、需要 `nvcc`，或某个第三方包明确要求时才需要安装 Toolkit。
- CUDA 11 之后支持同一大版本内的 minor version compatibility，但仍然要求驱动达到 NVIDIA 文档中的最低版本。CUDA 12.x 通常要求驱动版本 `>= 525`，CUDA 13.x 通常要求驱动版本 `>= 580`。更精确的表格以 NVIDIA 官方 CUDA Compatibility 文档为准。

## 3. 创建或进入虚拟环境

在项目根目录运行：

```powershell
cd D:\RCWA-3D-codex

# 如果还没有虚拟环境
python -m venv .venv

# PowerShell 激活
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

如果 PowerShell 拒绝执行激活脚本，可以在当前用户范围内放宽策略：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新打开终端并再次激活虚拟环境。

## 4. 安装 CUDA 版 PyTorch

打开 PyTorch 官方安装页：

<https://docs.pytorch.org/get-started/locally/>

选择：

- PyTorch Build: `Stable`
- Your OS: 按当前系统选择，例如 `Windows`
- Package: `Pip`
- Language: `Python`
- Compute Platform: 选择适合驱动的 CUDA 版本，通常优先选择官网提供的最新 CUDA 版本

然后复制网页生成的安装命令执行。

本项目只强依赖 `torch`，`torchvision` 和 `torchaudio` 不是必须项。若官网给出的 CUDA wheel index 是 `cu128`，可按下面形式安装：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

如果官网选择器给出的是其他 CUDA 版本，例如 `cu124`、`cu126` 或后续版本，请使用官网生成的命令，不要固定照抄上面的 `cu128`。

## 5. 安装本项目

在项目根目录、同一个虚拟环境中执行：

```powershell
python -m pip install -e ".[gpu]"
```

说明：

- PowerShell 中建议给 `".[gpu]"` 加引号。
- `.[gpu]` 会安装项目本体和 `torch>=2.0`。如果已经手动安装过 CUDA 版 PyTorch，`pip` 通常会复用现有版本。
- 如果之后发现 `torch.cuda.is_available()` 为 `False`，优先重新安装 CUDA 版 PyTorch，而不是修改项目代码回退 CPU。

## 6. 验证 CUDA 是否可用

先检查驱动和显卡：

```powershell
nvidia-smi
```

再检查当前 Python 环境中的 PyTorch：

```powershell
python -c "import torch; print('torch:', torch.__version__); print('torch cuda runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

期望结果：

- `cuda available: True`
- `torch cuda runtime:` 后面有版本号，例如 `12.8`
- `device:` 显示 NVIDIA GPU 名称

最后检查项目后端：

```powershell
python -c "import rcwa3d_isotropic as rcwa; b = rcwa.resolveBackend('cuda'); print(b.name, b.isCuda, b.device)"
```

期望输出类似：

```text
cuda True cuda
```

## 7. 运行测试和示例

基础测试：

```powershell
python -m unittest discover -s tests
```

快速运行一个 CUDA 示例：

```powershell
python examples\isotropic_example\homogeneousSlab.py
```

运行当前重点示例：

```powershell
python examples\isotropic_example\photonicCrystalSlab.py
python examples\isotropic_example\rectangularMetasurface.py
python examples\isotropic_example\applsci2019AuSiMetagratingField.py
```

这些示例文件中应保留：

```python
BACKEND = "cuda"
```

不要改成 `"cpu"`、`"numpy"` 或 `"torch-cpu"`。两个求解器都会主动拒绝 CPU 后端，以免计算时悄悄切到另一条数值路径。

## 8. 常见问题

### `RuntimeError: the isotropic solver requires a CUDA-enabled torch installation and visible CUDA device`

原因通常是：

- 当前环境安装了 CPU 版 PyTorch。
- 当前 Python 环境不是你安装 CUDA PyTorch 的环境。
- NVIDIA 驱动不可用或版本过旧。
- 没有可见 NVIDIA GPU。

处理：

```powershell
where python
python -m pip show torch
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

确认环境后，按第 4 节重新安装 CUDA 版 PyTorch。

### `ValueError: the ... solver is CUDA-only; use backend='cuda'`

这是预期行为。公开求解器不支持 CPU 后端：

```python
BACKEND = "cuda"
```

如果只是想做 CPU 对照，应单独设计内部参考脚本，不要在公开示例中改后端。

### `torch.cuda.is_available()` 是 `False`

按顺序检查：

1. `nvidia-smi` 是否能看到 GPU。
2. `python -m pip show torch` 是否来自当前虚拟环境。
3. `torch.__version__` 是否带 CUDA 标记，例如 `+cu128`。
4. `torch.version.cuda` 是否不是 `None`。
5. 是否在 IDE 中选择了同一个 `.venv` 解释器。

### CUDA out of memory

RCWA 的内存压力主要来自谐波阶数、层数、频谱点数和场分布采样网格。处理方式：

- 降低 `ORDER` 或二维阶数。
- 减小几何离散采样网格。
- 减少频谱扫描点数。
- 分批计算波长或频率。
- 关闭其他占用 GPU 显存的程序。

### 安装后 IDE 仍然报错

常见原因是 IDE 使用了另一个 Python 环境。请在 IDE 中选择项目虚拟环境：

```text
D:\RCWA-3D-codex\.venv\Scripts\python.exe
```

切换解释器后重启 IDE 或 Python kernel。

## 9. 维护建议

- 示例统一使用 `BACKEND = "cuda"`。
- 不要在示例里加入 CPU fallback；失败应直接暴露环境问题。
- 变更 CUDA / PyTorch 后，先运行第 6 节验证命令，再运行测试。
- 写场分布示例时，应显式定义用于展示的波长或频率，例如 `FIELD_WAVELENGTH`，避免扫描参数和场展示参数混淆。

## 10. 官方参考

- PyTorch 本地安装选择器：<https://docs.pytorch.org/get-started/locally/>
- NVIDIA CUDA Compatibility：<https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
