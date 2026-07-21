# AIMA AMD395 Qwen3.6 35B Linux 推理引擎

这是针对 AMD Ryzen AI Max+ 395 / Radeon 8060S（`gfx1151`）专项优化的
`Qwen3.6-35B-A3B-BF16` batch-1 推理引擎。v1.0.0 提供模型常驻、确定性
OpenAI Chat Completions 子集、CLI 操作和精确前缀缓存。

仓库包含完整引擎源码、原生 provider 源码与已验证二进制、服务端、CLI
和复现元数据；不包含模型权重，也不包含约 69.3 GB 的启动镜像。

## 适用范围

- 硬件：AMD Ryzen AI Max+ 395、Radeon 8060S、96 GB 统一内存；
- 系统：Linux x86-64，验证环境为 Ubuntu 24.04、ROCm 7.2.1；
- 模型：指定 index hash 的 Qwen3.6-35B-A3B BF16 checkpoint；
- 模式：batch 1、`temperature=0`、`top_p=1`；
- 总上下文：262,144 token，有效输入上限为 `262144 - max_tokens`；
- 前缀缓存：一个 entry，最多 32,768 个输入 token；
- v1.0.0 不支持 streaming、tools、并发 batch 或随机采样。

超出范围的请求会明确失败，不会静默切换到未经验证的模式。

## 快速使用

```bash
git clone https://github.com/approaching-ai/aima-amd395-qwen36-35b-linux-engine.git
cd aima-amd395-qwen36-35b-linux-engine
./aima-engine verify

export AIMA_RUNTIME_PYTHON=/path/to/rocm-vllm-venv/bin/python
export AIMA_MODEL_DIR=/path/to/Qwen3.6-35B-A3B

./aima-engine prepare-images \
  --model-dir "$AIMA_MODEL_DIR" \
  --lane0-dir /mnt/nvme0/aima-qwen36 \
  --lane1-dir /mnt/nvme1/aima-qwen36 \
  --state-dir "$HOME/.cache/aima-qwen36" \
  --output-manifest "$HOME/.config/aima-qwen36/striped-image-manifest.json"

export AIMA_IMAGE_MANIFEST="$HOME/.config/aima-qwen36/striped-image-manifest.json"
./aima-engine doctor
./aima-engine serve --host 127.0.0.1 --port 8000
```

另一个终端中：

```bash
./aima-engine status
./aima-engine chat --max-tokens 128 "解释前缀缓存为什么能降低 TTFT。"
./aima-engine shutdown
```

完整安装、API、性能和安全说明见英文主 README 中的文档索引。

## 已验证性能

- q8192/output512 冷启动请求：prefill `1591` tok/s，decode `32.12` tok/s；
- q8192/output1024：prefill `1587` tok/s，decode `32.18` tok/s；
- 命令启动到服务 ready：中位数 `27.31` 秒；
- q32768 近完整前缀命中：TTFT 中位加速 `110.1x`，decode 保持率最低
  `0.9997`；
- 正确性：KLD `0.0002768`、top-1 `1.0`、128 token 完全一致；
- HTTP usage、自动停止、strict/exact prefix cache：`76/76` 通过。

原始 D275 衰减比例仍是未完全达到的工程目标，不应误写为已完成；当前发布
通过的是项目定义的全部阻塞性能下限。

## 许可证

项目代码采用 Apache License 2.0。AMD 生成的 CK-Tile 文件保留 MIT
许可证。模型权重不随仓库分发，也不会被本项目重新授权。
