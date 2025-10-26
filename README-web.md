# MixTeX Web 应用部署指南

本文档旨在说明如何部署和运行 MixTeX 的 Web 版本，包括本地直接运行和通过 Docker 容器运行两种方式。

## 功能特性

- **Web 界面**: 提供一个用户友好的 Web 界面，支持拖拽、粘贴和批量上传图片进行 LaTeX 识别。
- **GPU 加速**: 自动检测并优先使用 NVIDIA GPU (CUDA) 进行加速，同时提供手动切换至 CPU 的选项。
- **Docker 支持**: 提供 `Dockerfile`，支持快速、跨平台的容器化部署。
- **自动化构建**: 集成 GitHub Actions，可自动构建多架构 (`linux/amd64`, `linux/arm64`) 的 Docker 镜像并发布到 Docker Hub。

## 环境准备

在开始之前，请确保您的环境中已安装以下软件：

- Python 3.8+ 和 Pip
- [Docker](https://www.docker.com/get-started)
- (可选，若使用 GPU) [NVIDIA 显卡驱动](https://www.nvidia.com/Download/index.aspx) 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 本地运行

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/haohaoget/MixTeX-Latex-OCR.git
cd MixTeX-Latex-OCR
```

### 步骤 2: 下载并放置模型文件

Web 应用需要 ONNX 模型文件才能运行。请下载模型文件，并将其解压后的 `onnx` 文件夹放置在项目的根目录下。

最终的目录结构应如下所示：

```
MixTeX-Latex-OCR/
├── onnx/
│   ├── encoder_model.onnx
│   ├── decoder_model_merged.onnx
│   └── ... (其他模型相关文件)
├── app.py
├── Dockerfile
└── ... (其他项目文件)
```

### 步骤 3: 安装依赖

我们为 Web 应用提供了一个独立的依赖文件 `requirements.app.txt`。

```bash
pip install -r requirements.app.txt
```

### 步骤 4: 运行应用

```bash
streamlit run app.py --server.port 3399
```

应用启动后，您可以在浏览器中打开 `http://localhost:3399` 进行访问。

## 使用 Docker 部署

使用 Docker 是推荐的部署方式，它可以隔离环境，简化部署流程。

### 步骤 1: 构建 Docker 镜像

在项目根目录下，运行以下命令构建镜像：

```bash
docker build -t mixtex-web .
```

### 步骤 2: 运行 Docker 容器

运行容器时，您需要通过 `-v` 参数将本地的模型文件夹挂载到容器内部的 `/app/onnx` 路径。

**使用 CPU 运行:**

```bash
docker run -p 3399:3399 -v /path/to/your/onnx:/app/onnx mixtex-web
```
> **注意**: 请将 `/path/to/your/onnx` 替换为您本地存放 `onnx` 文件夹的 **绝对路径**。

**使用 NVIDIA GPU 运行:**

如果您已安装 NVIDIA Container Toolkit，可以使用 `--gpus all` 参数来启用 GPU 加速。

```bash
docker run --gpus all -p 3399:3399 -v /path/to/your/onnx:/app/onnx mixtex-web
```

## 自动化构建与发布

本项目已配置 GitHub Actions 工作流 (`.github/workflows/docker-publish.yml`)。当代码被推送到 `main` 分支时，它会自动执行以下操作：

1.  构建支持 `linux/amd64` 和 `linux/arm64` 架构的 Docker 镜像。
2.  将镜像推送到 Docker Hub。

要启用此功能，您需要在您的 GitHub 仓库的 `Settings` -> `Secrets and variables` -> `Actions` 中设置以下两个密钥：

- `DOCKERHUB_USERNAME`: 您的 Docker Hub 用户名。
- `DOCKERHUB_TOKEN`: 您的 Docker Hub 访问令牌。
