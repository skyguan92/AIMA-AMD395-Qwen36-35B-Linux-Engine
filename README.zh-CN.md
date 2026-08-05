# AIMA AMD395 Qwen3.6 35B Linux 推理引擎

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.5.0-green.svg)](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.0)
[![Hardware](https://img.shields.io/badge/GPU-gfx1151-orange.svg)](docs/INSTALL.md)

这是一个面向 AMD Ryzen AI Max+ 395 / Radeon 8060S 的 batch-1
`Qwen3.6-35B-A3B-BF16` 专用推理引擎。

v1.5.0 提供真正的 SSE 流式输出与 OpenAI function tools，同时保持可搬移原生
运行包：运行时不加载 Python、PyTorch、vLLM、
Triton、Transformers，也不依赖宿主机安装 ROCm userspace。发布包内含静态
启动器、原生引擎、固定版本的 ROCm/AOTriton/CK 动态库、glibc loader、许可证
和资格验证记录。模型权重不随项目分发。

> **版本边界：**v1.4.0 新增 `doctor`、`--build-info`、bearer 鉴权、socket
> 超时和加固后的 systemd 模板；v1.4.1 新增变长 cold prompt 与普通多轮
> cache miss 回退；v1.5.0 新增常驻 q1024/q2048/q4096/q8192 prefill 调度与
> 容量受限的多条目 prefix LRU。更早版本仍遵循其文档中的请求延迟边界。

English: [README.md](README.md)

## 作者与仓库关系

本项目由
[关嘉伟 / Jiawei Guan（@skyguan92）](https://github.com/skyguan92)
创建并维护。

- **原始上游：** [skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine)
- **组织 fork 与官网主展示版本：** [Approaching-AI/AIMA-AMD395-Qwen36-35B-Linux-Engine](https://github.com/Approaching-AI/AIMA-AMD395-Qwen36-35B-Linux-Engine)

Python 包元数据和引用文件使用同一个、可由 GitHub 识别的个人作者身份；现有版权
声明保持不变。
发布资产与 CI 由个人原始上游发布；组织 fork 作为稳定的公开展示与 issue 入口。
产品改动在两个仓库间保持同步，组织专属的身份元数据可以不同。

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

HTTP prompt 可以是任意正 token 长度，只要 prompt 与请求输出总和不超过配置的
cache capacity。所选上下文是高性能 AOT prefill 专用长度：较短 cache miss 使用
正确的常驻逐 token 回退，较长 miss 先执行 AOT 前缀，再只逐 token 执行尾部。
Prefix hit 只影响时延，不再决定请求能否执行。绝对窗口上限仍为 262,144 token。
原生版本现已替代 v1.1 的公开性能矩阵；Python 版本仅保留为兼容与来源参考。
机器可读边界见
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

先从[个人上游 v1.5.0 Release](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.0)
下载运行包与校验文件：

```bash
sha256sum -c aima-engine-native-portable-*.tar.zst.sha256
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
    "messages": [{"role": "user", "content": "你好"}],
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
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

同一接口也支持 OpenAI function `tools`、`tool_choice`、
`parallel_tool_calls`、assistant 工具调用历史以及 tool 响应。完整请求与返回示例、
变长 prompt 执行规则见 [docs/API.md](docs/API.md)。

前台运行时可用 `Ctrl-C` 或 `SIGTERM` 关闭，也可以：

```bash
curl -fsS -X POST http://127.0.0.1:8000/shutdown
```

发布包的 `share/systemd/` 提供 service 和环境变量模板。安装后可直接用
`systemctl start|status|stop aima-engine` 管理常驻服务。

## 原生 CLI

已发布的 v1.5.0 CLI 提供：

```text
aima-engine --build-info
aima-engine doctor [--model-dir PATH] [--device INDEX] [--json]
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
export AIMA_API_KEY_FILE=/path/to/client-readable-api-key
aima-engine models
aima-engine chat --stream "PROMPT"
aima-engine chat --stream --tools-json tools.json --tool-choice auto "PROMPT"
aima-engine chat --messages-json conversation.json --tools-json tools.json
```

纯 Python wheel 有意保持为无运行时依赖的客户端，只暴露 `status`、`models`、
`chat` 和 `shutdown`。旧的 Python 服务端/镜像管理命令只在完整源码 checkout
中出现；实际部署使用单独验证过的原生运行包。可通过 `--api-key-file` 或
`AIMA_API_KEY_FILE` 提供 bearer token，不需要把 token 放进进程参数。

## 原生成品性能

下表来自最终原生二进制在 AMD395 目标机上的结果。协议为同配置三次取中位数；
昂贵 cell 若前两次差异不超过 3%，两次即可。

| 输入 | output512 prefill | output512 decode | output1024 prefill | output1024 decode |
|---:|---:|---:|---:|---:|
| 1,024 | 1630 | 34.00 | 1630 | 33.99 |
| 2,048 | 1685 | 33.86 | 1685 | 33.86 |
| 4,096 | 1572 | 33.26 | 1572 | 33.25 |
| 8,192 | 1656 | 32.29 | 1656 | 32.28 |
| 16,384 | 1438 | 30.79 | 1438 | 30.79 |
| 32,768 | 1365 | 28.23 | 1365 | 28.23 |
| 65,536 | 1176 | 24.68 | 1176 | 24.68 |
| 131,072 | 868.2 | 19.60 | 868.2 | 19.60 |

最大窗口端点分别达到：262143/output1 prefill `556.5` token/s，
261632/output512 为 `560.5 / 14.05` prefill/decode token/s，
261120/output1024 为 `535.8 / 14.04`。19 个 cell 全部达到冻结基线的
97%；最低 prefill/decode 保留率分别是 `1.013x` 与 `0.9858x`。相对
v1.4.1 完整矩阵，最差 prefill/decode 中位数变化为 `-2.259%` 与
`-0.1280%`，均在 3% 协议范围内。

其他门槛：

- 9 个上下文直至 q261632 的全词表 KLD 全部小于 `0.005`，最大值
  `0.002174`，top-1 全一致；
- 冻结 q8192 fixture 的 128-token 输出逐 token 完全一致；
- 冻结 answer-only MMLU-256 回归得到 `216/256`（`84.375%`），与 GB10
  vLLM 参考分数完全相同；256 个 prompt-token 哈希全部一致，其中 250 个
  completion-token 哈希逐 token 完全相同；
- q8192 command-to-ready 中位数 `51.16 s`，低于 `51.41 s` 上限；
- q32768 exact-prefix TTFT 加速 `2626x`，decode 保留率 `1.0001`；
- HTTP 两次请求期间模型装载次数始终为 1，第二次命中 exact cache，并可干净关闭；
- chunked SSE 与非流式的 token/text 哈希一致，stream/non-stream 工具调用一致，
  客户端断连后服务仍健康。
- 16-token cold prompt、exact replay、36-token 普通下一轮用户请求，以及长上下文
  后的无关短请求全部通过；独立会话保持隔离并返回 HTTP 200；
- q1024/q2048/q4096/q8192 raw-token 请求都选中了对应的常驻 AOT bucket，
  A/B/A 请求序列验证了 4 条目 LRU 的复用。

完整精度、每次测量值和组件哈希会在发布后镜像到
`benchmarks/results/native-portable-product-v1.5.0.json`，同时随包保存为
`share/aima/qualification.json`。

## 从源码构建

运行时不需要框架，构建时仍需要工具链。合格 builder 需要 ROCm/HIP、用于确定性
代码生成的 Python、固定 commit
`6667a9021713f794a2c9aee4696c19f6cf376235` 的 AMD Composable Kernel，
以及固定 AOTriton 0.11.1 的 headers/library/image：

```bash
export CK_DIR=/path/to/composable-kernel
export AOTRITON_ROOT=/path/to/distribution/root/containing/include-and-lib
export QUALIFICATION_RECORD=/path/to/qualified-product-result.json
export AIMA_RELEASE_VERSION=X.Y.Z
export AIMA_RELEASE_TAG=vX.Y.Z

make check
make build-native build-native-runtime
# 按文档对这些精确产物完成资格验证。
make package-native
```

打包器会拒绝绝对 RUNPATH、未闭合 ELF 依赖以及任何与完整 qualification
哈希不一致的可执行文件/provider，收集第三方许可证，生成递归 SHA-256
manifest，并在 `dist/` 输出一个确定性的 `.tar.zst` 包。
打包步骤不会重新构建已经完成验证的产物。

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

HTTP 服务默认只监听 `127.0.0.1`，支持从 `--api-key-file` 读取 bearer
token；默认拒绝未鉴权的非 loopback 监听，并可通过
`--disable-http-shutdown` 移除停服接口。TLS、限流和多用户授权仍应由网关提供。

AIMA 项目代码采用 [Apache License 2.0](LICENSE)。随包第三方组件继续遵循各自
许可证，见 [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
及发布包的 `licenses/`。模型权重不包含在项目中。
