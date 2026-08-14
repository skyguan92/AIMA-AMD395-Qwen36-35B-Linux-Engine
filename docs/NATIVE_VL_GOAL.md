# Native VL product goal

> Status: proposed hard target
> Baseline: `v1.5.1`
> Last updated: 2026-08-14

## One-sentence goal

在 `v1.5.1` 产品化原生引擎上补齐固定
`Qwen3.6-35B-A3B-BF16` 模型的完整图片和视频理解能力，使支持范围、输入处理、
模型语义和 OpenAI-compatible 服务行为与固定原版模型及 vLLM 参考一致，同时
保持现有文本正确性、文本性能、262,144-token 窗口、常驻缓存、启动、部署和
安全能力不回退。

## Completion summary

下表五项必须同时通过，缺一项都不能称为 VL 产品完成。

| ID | Goal | Blocking acceptance |
|---|---|---|
| `G1` | 完整 VL 功能 | 原版模型的图片/视频能力，以及固定 vLLM 对该模型的 multimodal 请求行为全部支持；不能只做单图 MVP |
| `G2` | VL 正确性一致 | processor、vision boundaries、M-RoPE、full-vocabulary logits、greedy 输出和任务级评测全部通过固定门槛 |
| `G3` | 文本产品零回退 | `v1.5.1` 的正确性、19-cell 性能、启动、prefix cache、SSE、tools、长窗口和 MMLU 回归全部不下降 |
| `G4` | VL 性能达到高速引擎标准 | 同一 AMD395、同一输入和同一计时边界下，VL TTFT、吞吐和总时延逐格达到或超过固定 vLLM |
| `G5` | 可发布的原生产品 | 仍是单一原生常驻进程，无 Python 推理栈，96 GiB 内可运行，通过完整 package、security 和 release evidence 门槛 |

## 1. Frozen product baseline

### 1.1 Release identity

| Item | Frozen value |
|---|---|
| Product repository | `AIMA-AMD395-Qwen36-35B-Linux-Engine` |
| Product tag | `v1.5.1` |
| Release commit | `6f3e669ac897eaabfeceb7f193a5e02708a4d95e` |
| Embedded native source commit | `65c198415709dad6d046c247acab3dc9df2a95a0` |
| Qualified native engine SHA-256 | `a9f18771175757af080c8a1d8d7e3fb3906c9aa41b43a496686103b626f80262` |
| Product contract | `native/product-contract-v1.5.1.json` |

后续验收必须使用这份 release binary、发布证据和相同目标机重跑得到的配对数据。
不能换成较慢的历史版本或较弱的 floor 来制造“无回退”。

### 1.2 Model identity

| Item | Frozen value |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` |
| Revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Dtype | BF16 |
| Config SHA-256 | `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` |
| Checkpoint index SHA-256 | `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83` |
| Tokenizer SHA-256 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Tokenizer config SHA-256 | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |

模型 revision、processor、chat template 或 tokenizer 任一变化，都属于新的
qualification regime，不能沿用本目标的正确性结论。

### 1.3 VL reference identity

首个 oracle 前必须生成一个 hash-bound reference manifest，至少固定：

- vLLM `0.19.1rc1.dev300+g29e5d1020`；
- PyTorch `2.10.0+git8514f05`；
- Transformers `4.57.6`；
- AMD395 host facts、ROCm/driver、完整启动命令和全部环境变量；
- processor 参数、媒体数量限制、允许的媒体目录、视频 fps/帧采样参数；
- prompt、media、processor outputs、sampling 参数和原始响应 hashes。

VL reference 必须真正启用 multimodal 路径；`--language-model-only` 或
`--skip-mm-profiling` 的文本-only 进程不能充当 VL reference。以后若升级
vLLM，必须先完成新旧 reference 差分，且不能缩小本目标。

## 2. G1 — Full VL functional parity

“功能一致”定义为：固定原版模型明确支持的视觉能力，以及固定 vLLM 为该模型
提供的 multimodal 输入、处理和响应行为，本产品都必须实现。这里对齐的是 VL
surface，不是 vLLM 的全部通用 serving surface；vLLM 为其他模型提供的能力、
本模型不支持的 audio，以及与 VL 无关的调度功能不自动进入范围。

### 2.1 Required successful requests

| Surface | Required cases |
|---|---|
| Image | 单图、多图；不同格式、长宽比、方向、透明度和动态分辨率 |
| Video | 单视频、多视频；不同容器、fps、时长、帧数和采样位置 |
| Mixed media | 图片+视频；文本与媒体按所有合法顺序交错 |
| Conversation | system/user/assistant/tool 历史；单轮、多轮；媒体复用和替换 |
| OpenAI API | `/v1/chat/completions` 的 stream/non-stream content parts |
| Generation | 沿用产品现有 greedy 参数合同；VL 的 token、usage、finish reason 和 stream 行为与参考一致 |
| Tools | VL 请求中的 tools、tool choice、assistant/tool history 和结构化 tool call |
| Transport | 固定 vLLM 接受的 URL、data URI/base64、本地文件和媒体格式 |
| Residency | 同一进程连续处理 text/image/video/mixed 请求，不重新加载语言模型 |

真实边界值不能由实现方拍脑袋决定。必须通过 capability probe 从固定
processor/vLLM 生成 min/typical/max 和离散边界 manifest，包括最大媒体数量、
最大接受尺寸/帧数、media-token 预算及错误行为。产品支持范围不得小于该 manifest。

### 2.2 Required semantics

- chat template、special tokens、image/video placeholders 和 token 顺序一致；
- image resize/resample/normalize、patchify、spatial merge 和动态分辨率一致；
- video decode、frame sampling、temporal patch、fps/时间戳处理一致；
- `image_grid_thw`、`video_grid_thw`、media-token 数量和位置一致；
- vision encoder、merger、media embedding 注入和文本 embedding 拼接一致；
- M-RoPE position ids/delta、时间轴、prefill 和 KV-cache 续写一致；
- 媒体 token 与文本 token 共同遵守 `262144` 总窗口；
- 相同非法输入应得到兼容的 HTTP 状态、错误类别和安全行为。

### 2.3 Existing product behavior that must remain

新增 VL 后，`v1.5.1` 已有能力仍是同一产品的一部分：

- native tokenizer 和 Qwen chat/tool template；
- live HTTP/1.1 chunked SSE 与断连取消；
- OpenAI function tools、forced tool 和 assistant/tool history；
- bearer authentication、fail-closed remote bind、timeouts 和 shutdown policy；
- `doctor`、`--build-info`、health/models endpoints 和 systemd readiness；
- variable-length cold prefill、普通 multi-turn cache miss 和请求隔离。

### 2.4 VL cache correctness

- multimodal prefix key 必须绑定有序媒体内容 digest、processor 参数、媒体 token
  和文本 token，不能只比较相同的 image/video placeholder token；
- 相同媒体 A 可以复用，内容不同但 URL、文件名或尺寸相同的媒体 B 不能误命中；
- A/B/A、URL 内容变化、base64 与本地文件等价输入、视频采样参数变化必须有
  独立回归用例；
- 如果某类媒体无法建立安全稳定的 key，可以保守 miss，但不能错误复用；
- 任何 media/vision embedding cache 都只能改变时延，不能改变 logits、输出
  token、usage 或错误语义。

### 2.5 Scope boundary

本目标不要求顺带补齐与 VL 无关的 vLLM 通用功能。dynamic batching、并发执行、
multi-model serving、stochastic sampling、自定义 stop、deprecated functions 和
structured response formats 继续沿用 `v1.5.1` 的现有范围，除非另立产品目标。
这些边界不能用于删减图片、视频、mixed-media 或 multimodal conversation 能力。

## 3. G2 — VL correctness parity

VL 正确性必须由四层证据共同通过，不能只展示“图片问答看起来合理”。

| Gate | Required evidence | Pass condition |
|---|---|---|
| Processor | token ids、placeholder spans、grid THW、resize shape、frame indices、timestamps、pixel tensors | 离散值精确一致；浮点容差在首跑前冻结，不得事后扩大 |
| Vision boundary | patch embed、vision blocks `0/13/26`、merger、injected embeddings | cosine `>=0.999` 且 relL2 `<=0.002`；shape/index 精确一致 |
| Language boundary | M-RoPE ids/delta、首个语言层、最终 norm 和 lm_head | cosine `>=0.999` 且 relL2 `<=0.002`；离散状态精确一致 |
| End-to-end logits | image、video、multi-image、multi-video、mixed-media 的 full-vocabulary teacher-forced rows | 每个位置 `KLD<0.005`，top-1 `=1.0` |
| Deterministic generation | 冻结 greedy fixtures | token ids、tool calls、finish reason 和 usage 精确一致 |
| Task quality | 冻结 image 与 video regression suites | candidate 分数分别不得低于同一 corpus 上的固定 vLLM |
| Error parity | 损坏、空、超限、超时、不可访问和类型不匹配输入 | 接受/拒绝、HTTP 状态与错误类别兼容 |

正确性 corpus 至少覆盖单图、多图、单视频、多视频、mixed media、多轮、tools、
stream、最大合法边界和每种错误类别。只测英文、单一分辨率、单图或短视频不够。

## 4. G3 — Text product no regression

### 4.1 Correctness and capability

VL 版本必须用同一 release protocol 重跑并保留：

- 9 个上下文到 q261632 的 full-vocabulary `KLD<0.005`，top-1 全部一致；
- q8192 冻结 fixture 的 128-token 输出精确一致；
- MMLU-256 不低于当前 `218/256`，invalid answers 仍为 `0`；
- 256 个冻结 prompt-token hashes 全部一致；
- native HTTP conformance、SSE/non-stream token/text parity、stream/non-stream
  tool parity、tool history、client disconnect 和服务健康全部通过；
- text-only 请求不运行 media processor、vision kernels，也不分配 request-level
  media scratch。

### 4.2 Frozen 19-cell performance matrix

下表单位为 tok/s。candidate 必须与 `v1.5.1` release binary 在同一 AMD395
上做交替配对测量。

| Input | output512 prefill | output512 decode | output1024 prefill | output1024 decode |
|---:|---:|---:|---:|---:|
| 1,024 | 1630 | 34.00 | 1630 | 34.02 |
| 2,048 | 1693 | 33.85 | 1693 | 33.85 |
| 4,096 | 1569 | 33.32 | 1569 | 33.30 |
| 8,192 | 1660 | 32.30 | 1660 | 32.28 |
| 16,384 | 1440 | 30.79 | 1440 | 30.78 |
| 32,768 | 1358 | 28.22 | 1358 | 28.22 |
| 65,536 | 1170 | 24.65 | 1170 | 24.65 |
| 131,072 | 869.7 | 19.62 | 869.7 | 19.62 |

最大窗口请求也必须保留：

| Input/output | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 262143/1 | 555.2 | n/a |
| 261632/512 | 555.1 | 14.04 |
| 261120/1024 | 559.3 | 14.02 |

### 4.3 No-regression decision rule

- throughput：每个请求格子的 candidate 配对中位数必须 `>=1.000x` 当前
  `v1.5.1`；latency：必须 `<=1.000x`；
- 至少 5 组 release/candidate 交替配对；若噪声导致结论不确定，增加重复，
  不能把噪声当作允许回退的预算；
- 旧 product contract 的 `0.97x` 独立 floor 继续作为安全底线，但不是本次
  VL 改动可消费的 3% 回退额度；
- q8192 command-to-ready 中位数不得高于当前 `44.90 s`；
- `READY=1` 必须表示语言模型和 VL 路径都已可服务；不能提前报告 text-ready，
  再把 vision 权重加载或初始化时延转移到首个 VL 请求；
- q32768 exact-prefix TTFT speedup 不低于 `2637x`，decode retention 不低于
  `1.0003x`；
- q8192 的四条 prefix-LRU entries、A/B/A 恢复、variable-length composed AOT
  prefill 和 zero serial cold-tail 行为不变；
- 96 GiB 内仍支持全部 standard contexts、三个最大窗口端点和现有 KV/cache
  容量，新增 vision residency 不得挤掉文本产品容量；
- 每个结论绑定 exact binary、commit、host facts、命令、缓存状态、原始输出
  和 correctness result，aggregate 不能掩盖单格回退。

## 5. G4 — Native VL performance

VL 性能比较必须在同一 AMD395、同一模型、同一 processor 输出、同一媒体、
同一有效 context、同一 output length、同一 cache/warmup 和同一计时边界下进行。

### 5.1 Required performance surfaces

| Surface | Minimum coverage |
|---|---|
| Image | 单图 min/typical/max；多图 typical/max；多长宽比 |
| Video | 单视频 min/typical/max frames；多视频；采样边界 |
| Mixed | image+video；text/media interleave；multi-turn |
| Context | short、1k、8k、32k、128k、接近 262144 总 token 边界 |
| Output | 1、512、1024 |
| Cache | cold media、warm media、cache disabled、media exact hit、A/B/A |

矩阵不要求所有维度做完整笛卡尔积，但必须覆盖每项 min/typical/max、processor
离散跳变边界和 pairwise 组合。确切格子由 reference capability manifest 生成。

### 5.2 Required metrics and thresholds

- 分开记录 media fetch/decode、processor、vision encode、LLM prefill、decode、
  TTFT、total latency、throughput、host RSS 和 peak GTT；
- 每个可比较格子的 candidate TTFT 和 total latency 配对中位数不得高于
  vLLM，vision/prefill/decode throughput 不得低于 vLLM；
- media embedding 进入语言模型后必须复用现有 native fast path；相同有效
  token 数下的 decode throughput 不得低于对应文本产品曲线；
- candidate 与 vLLM 必须使用对称的媒体下载、预处理、cache 和 warmup；不能
  把 candidate 的前处理移出计时窗口；
- 如果固定 vLLM 在某个完整能力格子上因资源不足无法运行，该格子标记为
  `reference unavailable`，不能自动算 candidate 通过，也不能删除该产品能力；
- 性能验收遵守与文本相同的至少 5 组交替配对和逐格判定规则。

## 6. G5 — Native product and release boundary

完成后的产品仍必须满足 `v1.5.1` 的部署原则：

- 一个常驻 native engine 进程完成 media processor、vision encoder 和 LLM；
- language 与 vision 权重、必要的派生布局和固定 runtime state 在 `READY=1`
  前完成校验与常驻初始化，正常请求期间不读 checkpoint/oracle；
- runtime 不依赖 Python、PyTorch、vLLM、Triton JIT、Transformers 或 host
  ROCm userspace；
- 标准 Hugging Face Safetensors 仍是独立下载的模型输入，模型权重不进包；
- source 保持一个参数化 vision stack、一个参数化 language layer 和共享循环，
  不生成 per-layer source files；
- 完整 native userspace、vision kernels、licenses、doctor checks、product
  contract、qualification 和 recursive manifest 进入可搬移 archive；
- media URL/local-file 访问有显式 allowlist、大小/时长限制、超时、重定向和
  SSRF/path traversal 防护；鉴权和日志不能泄露媒体或凭据；
- `make check`、`make security-scan`、`make verify-evidence`、isolated bundle、
  第二台 AMD395、长时间 resident mixed-workload 和 rollback 验证全部通过；
- release tag 不可移动，qualification 必须绑定 exact source commit、binary
  hashes、commands 和 raw artifacts。

## 7. Required delivery artifacts

实现工作必须最终交付以下机器可核验材料：

1. `vl-reference-manifest.json`：冻结原模型、processor、vLLM、环境和命令；
2. `vl-capability-manifest.json`：所有成功/失败请求形态及准确边界；
3. `vl-oracle-manifest.json`：processor、boundary 和 end-to-end oracle hashes；
4. `vl-api-render-manifest.json`：冻结完整 API capability corpus 的真实 OpenAI
   HTTP prompt token vectors、placeholder spans 和 structured-output 配置；
5. `vl-serving-render-manifest.json`：冻结数值 oracle serving corpus 的真实 HTTP
   prompt vectors，并与私有 processor-to-logits prompt 独立绑定；
6. `vl-correctness.json`：逐 fixture KLD/top-1/token/task-score 结果；
7. `vl-performance.json`：逐格 paired runs、分阶段耗时、内存与判定；
8. `text-v151-nonregression.json`：完整文本 product requalification；
9. 更新后的 machine product contract、package qualification、evidence archive
   和 checksum sidecar；
10. API、安装、架构、性能、安全、限制和回滚文档。

这些文件的最终路径由实现阶段按现有 `benchmarks/results/`、
`benchmarks/runs/` 和 `share/aima/` 约定确定；目标和门槛以本文为准。

## 8. What does not count as done

以下任一情况都只能算中间原型：

- 只支持单张图片，或者没有视频、多媒体、交错和多轮；
- 通过 Python/Transformers/vLLM sidecar 完成 media processor 或 vision encode；
- 通过缩小媒体数量、分辨率、视频帧数或 262,144-token 窗口来通过；
- 文本变慢，但仍勉强高于历史 `0.97x` floor；
- 只测 cache hit、预热输入或把 media preprocessing 排除出 candidate 计时；
- 只有示例答案，没有 processor、boundary、logits 和任务级 correctness；
- 因 vLLM 某个大格子 OOM 就把对应产品能力删除或自动判定性能通过；
- 牺牲四条 prefix cache、最大窗口、SSE、tools、安全或可搬移部署；
- 有实现代码但没有完整 hash-bound qualification 和 release evidence。

## Decision record

- `2026-08-13` — 目标必须基于产品仓 `v1.5.1` 修订，而不是另起一个弱于
  当前产品合同的实验引擎目标。
- `2026-08-13` — 最终范围是固定原版模型与固定 vLLM 对该模型的完整视觉/VL
  能力；image-only 或缩小 envelope 不能作为完成态。
- `2026-08-13` — 正确性和性能均为阻断式 no-regression gate；文本 floor 是
  当前 native product，VL performance floor 是同机固定 vLLM。
- `2026-08-14` — 私有 processor-to-logits prompt 不能替代真实 OpenAI HTTP
  render 边界；两类证据必须分别捕获、hash-bound 并在 native qualification 中
  同时验证。

## Source-of-truth pointers

- Product contract: `native/product-contract-v1.5.1.json`
- Current API: `docs/API.md`
- Native architecture: `docs/ARCHITECTURE.md`
- Published performance and correctness: `docs/PERFORMANCE.md`
- Release identity and qualification flow: `docs/RELEASE.md`
- Public product summary: `README.md` and `README.zh-CN.md`
