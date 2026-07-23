# AIMA AMD395 Qwen3.6 35B Linux 推理引擎

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.3.0-green.svg)](CHANGELOG.md)
[![Hardware](https://img.shields.io/badge/GPU-gfx1151-orange.svg)](docs/INSTALL.md)

这是一个面向 AMD Ryzen AI Max+ 395 / Radeon 8060S 的 batch-1
`Qwen3.6-35B-A3B-BF16` 专用推理引擎。

v1.3 提供真正的 SSE 流式输出与 OpenAI function tools，同时保持可搬移原生
运行包：运行时不加载 Python、PyTorch、vLLM、
Triton、Transformers，也不依赖宿主机安装 ROCm userspace。发布包内含静态
启动器、原生引擎、固定版本的 ROCm/AOTriton/CK 动态库、glibc loader、许可证
和资格验证记录。模型权重不随项目分发。

English: [README.md](README.md)

## 先看清楚支持边界

原生便携版本已通过完整 batch-1 发布矩阵：

| 输入 token | 输出 token | 状态 |
|---:|---:|---|
| 1,024 | 512 / 1,024 | 已验证 |
| 2,048 | 512 / 1,024 | 已验证 |
| 4,096 | 512 / 1,024 | 已验证 |
| 8,192 | 512 / 1,024 | 已验证 |
| 16,384 | 512 / 1,024 | 已验证 |
| 32,768 | 512 / 1,024 | 已验证 |
| 65,536 | 512 / 1,024 | 已验证 |
| 131,072 | 512 / 1,024 | 已验证 |
| 262,143 | 1 | 最大窗口端点已验证 |
| 261,632 | 512 | 最大窗口端点已验证 |
| 261,120 | 1,024 | 最大窗口端点已验证 |

HTTP cold prompt 经 chat template 编码后必须正好等于所选固定上下文。只有当更长
请求严格延续已缓存 token 前缀时，才会走 prefix extension。输入与输出总长度不得
超过 262,144 token。原生版本现已替代 v1.1 的公开性能矩阵；Python 版本仅保留为
兼容与来源参考。机器可读边界见
[native/product-contract.json](native/product-contract.json)。

## 运行环境

部署机器只需要：

- Linux x86-64，内核具备 AMDGPU/KFD 与 render device；
- Radeon 8060S / `gfx1151`；
- 128 GB 内存，并按文档设置 96 GiB GTT pool；
- 用户自行取得且哈希匹配的 26-shard BF16 Safetensors 模型。

不需要系统 ROCm 和 Python 虚拟环境。跨软件版本兼容来自随包 loader 与动态库；
解压后约 366 MiB，`.tar.zst` 压缩包约 101 MiB。Linux 内核驱动和 GPU
架构本身无法打包进去。

加载模型前先完成内存设置：
[中文](docs/MEMORY.zh-CN.md) · [English](docs/MEMORY.md)。

## 快速启动

```bash
tar --zstd -xf aima-engine-native-portable-*.tar.zst
cd aima-engine-native-portable-*

./bin/aima-engine --version
./bin/aima-engine serve \
  --model-dir /srv/models/Qwen3.6-35B-A3B \
  --context-tokens 8192 \
  --host 127.0.0.1 \
  --port 8000
```

进程启动时只装载一次模型，校验 69,321,221,376 字节有效权重，并让权重、执行计划、
KV/递归状态和 prefix cache 一直常驻。ready 时会输出一行 JSON。

另开终端检查：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

调用 OpenAI 兼容子集：

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [{"role": "user", "content": "编码后长度正好命中所选固定上下文的提示词"}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512
  }'
```

真正的逐 token SSE 流式输出：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aima-amd395-qwen36-35b",
    "messages": [{"role": "user", "content": "编码后长度正好命中固定上下文的提示词"}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

同一接口也支持 OpenAI function `tools`、`tool_choice`、
`parallel_tool_calls`、assistant 工具调用历史以及 tool 响应。完整请求与返回示例、
静态上下文计数规则见 [docs/API.md](docs/API.md)。

前台运行时可用 `Ctrl-C` 或 `SIGTERM` 关闭，也可以：

```bash
curl -fsS -X POST http://127.0.0.1:8000/shutdown
```

发布包的 `share/systemd/` 提供 service 和环境变量模板。安装后可直接用
`systemctl start|status|stop aima-engine` 管理常驻服务。

## 原生 CLI

```text
aima-engine --version
aima-engine serve --model-dir PATH --context-tokens N
aima-engine resident-session-probe --model-dir PATH [验证参数]
aima-engine tokenizer-probe --model-dir PATH --text TEXT
aima-engine chat-template-probe --model-dir PATH --user TEXT
aima-engine chat-template-probe --model-dir PATH --request-json JSON
```

`serve` 有意以前台形式运行，便于 systemd、容器和其他 supervisor 管理。随包的
内部 probe 可在没有框架运行时的情况下复现公开的正确性和性能结果。

源码安装后的可选控制 CLI 也可以直接作为客户端：

```bash
aima-engine chat --stream "PROMPT"
aima-engine chat --stream --tools-json tools.json --tool-choice auto "PROMPT"
aima-engine chat --messages-json conversation.json --tools-json tools.json
```

## 原生成品性能

下表来自最终原生二进制在 AMD395 目标机上的结果。协议为同配置三次取中位数；
昂贵 cell 若前两次差异不超过 3%，两次即可。

| 输入 | output512 prefill | output512 decode | output1024 prefill | output1024 decode |
|---:|---:|---:|---:|---:|
| 1,024 | 1636 | 34.02 | 1636 | 34.00 |
| 2,048 | 1690 | 33.85 | 1690 | 33.83 |
| 4,096 | 1574 | 33.28 | 1574 | 33.27 |
| 8,192 | 1654 | 32.26 | 1654 | 32.26 |
| 16,384 | 1433 | 30.67 | 1433 | 30.66 |
| 32,768 | 1357 | 28.18 | 1357 | 28.17 |
| 65,536 | 1183 | 24.61 | 1183 | 24.61 |
| 131,072 | 871.4 | 19.53 | 871.4 | 19.53 |

最大窗口端点分别达到：262143/output1 prefill `554.1` token/s，
261632/output512 为 `550.3 / 13.96` prefill/decode token/s，
261120/output1024 为 `565.8 / 13.91`。19 个 cell 全部达到冻结基线的
97%；最低 prefill/decode 保留率分别是 `1.014x` 与 `0.9839x`。

其他门槛：

- 9 个上下文直至 q261632 的全词表 KLD 全部小于 `0.005`，最大值
  `0.002174`，top-1 全一致；
- 冻结 q8192 fixture 的 128-token 输出逐 token 完全一致；
- q8192 command-to-ready 中位数 `44.69 s`，低于 `51.41 s` 上限；
- q32768 exact-prefix TTFT 加速 `2612x`，decode 保留率 `1.0000`；
- HTTP 两次请求期间模型装载次数始终为 1，第二次命中 exact cache，并可干净关闭；
- chunked SSE 与非流式的 token/text 哈希一致，stream/non-stream 工具调用一致，
  客户端断连后服务仍健康。

完整精度、每次测量值和组件哈希见
[benchmarks/results/native-portable-product-v1.3.0.json](benchmarks/results/native-portable-product-v1.3.0.json)。

## 从源码构建

运行时不需要框架，构建时仍需要工具链。合格 builder 需要 ROCm/HIP、用于确定性
代码生成的 Python、固定 commit
`6667a9021713f794a2c9aee4696c19f6cf376235` 的 AMD Composable Kernel，
以及固定 AOTriton 0.11.1 的 headers/library/image：

```bash
export CK_DIR=/path/to/composable-kernel
export AOTRITON_ROOT=/path/to/distribution/root/containing/include-and-lib

make check
make package-native
```

打包器会拒绝绝对 RUNPATH 和未闭合 ELF 依赖，收集第三方许可证，生成递归
SHA-256 manifest，并在 `dist/` 输出一个确定性的 `.tar.zst` 包。

安装与构建细节见 [docs/INSTALL.md](docs/INSTALL.md)。

## 仓库结构

```text
native/                      原生引擎、AOT closure 与产品契约
benchmarks/shape-lab/native/ CK-Tile 源码与兼容性产物
benchmarks/results/          发布资格验证记录
scripts/                     构建、依赖闭包审计与打包工具
packaging/systemd/           常驻服务模板
docs/                        安装、API、内存、架构与性能文档
aima_engine/                 保留的 v1.1 兼容控制面
```

## 安全与许可证

HTTP 服务没有内建鉴权，并提供 `POST /shutdown`。默认只监听
`127.0.0.1`，不要直接暴露到不可信网络。

AIMA 项目代码采用 [Apache License 2.0](LICENSE)。随包第三方组件继续遵循各自
许可证，见 [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
及发布包的 `licenses/`。模型权重不包含在项目中。
