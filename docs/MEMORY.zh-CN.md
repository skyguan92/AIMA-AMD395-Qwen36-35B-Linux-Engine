# AMD395 统一内存配置

本引擎不是把模型放进 BIOS 预留的大块固定显存，而是在 128 GB Ryzen AI
Max+ 395 主机上使用 96 GiB AMDGPU GTT 统一内存池。BIOS 的 UMA 切分和
Linux 内核参数两部分都必须配置正确。

## 重启后的目标状态

| 项目 | 合格值 |
|---|---:|
| 物理统一内存 | 128 GB |
| Linux `MemTotal` | 约 125 GiB |
| 固定/可见 VRAM | 512 MiB（`536870912` 字节） |
| AMDGPU GTT 池 | 96 GiB（`103079215104` 字节） |
| TTM 页数上限 | `25165824` 个 4 KiB 页 |

已验收运行中，完整模型进程的内存分配峰值约为 73.2 GB，最大上下文还会
更高。因此 64 GB 机器无法运行本 BF16 版本。128 GB 机器如果被 BIOS
切成 64 GB 系统内存 + 64 GB 固定显存也不合格；这种设置通常只能得到
约 32 GiB GTT。

## 1. 调整 BIOS UMA 切分

这一步需要本地显示器键盘或可靠的带外管理。不同厂商的选项名称可能是
`UMA Frame Buffer Size`、`iGPU Memory` 或 `Dedicated Graphics Memory`。

1. 进入 BIOS/UEFI 设置。
2. 把固定 UMA/帧缓冲设置为 **512 MiB**。如果没有 512 MiB 选项，只有在
   下面的启动后检查确实得到 512 MiB 时，才使用最小值或 `Auto`。
3. 保存并重启。

不要保留 64 GB 固定显存切分。模型使用下一步配置的 GTT 池，而大块固定
显存会直接减少 Linux 可用内存。

重启后检查系统内存和固定显存：

```bash
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo

for file in /sys/class/drm/card*/device/mem_info_vram_total; do
  test -r "$file" && printf '%s=' "$file" && cat "$file"
done
```

预期结果：`MemTotal` 约 125 GiB，且
`mem_info_vram_total=536870912`。DRM 卡号可能随机器或启动变化，因此应使用
通配符，不要写死 `card0`。

## 2. 配置 96 GiB GTT

在已验证的 Ubuntu 环境中，把下面两个参数加入现有 GRUB 内核命令行：

```text
ttm.pages_limit=25165824 amdgpu.gttsize=98304
```

`98304` 是以 MiB 表示的 96 GiB；`25165824` 是相同容量对应的 4 KiB 页数。

使用 `sudoedit` 编辑 `/etc/default/grub`，保留机器原有的其他启动参数：

```bash
sudoedit /etc/default/grub
```

例如：

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=25165824 amdgpu.gttsize=98304"
```

更新启动菜单并重启：

```bash
sudo update-grub
sudo reboot
```

这两个参数会影响整台机器的 AMDGPU 内存管理。不要在物理统一内存不足
128 GB 的机器上应用。内存检查通过也不代表任意内核版本都已得到性能
认证；v1.0.0 的合格内核和 ROCm 版本见 [`INSTALL.md`](INSTALL.md)。

## 3. 授予 GPU 设备权限

运行服务的实际账户必须同时属于 `render` 和 `video`：

```bash
sudo usermod -aG render,video "$(id -un)"
```

随后完整退出并重新登录，或直接重启。`/etc/group` 中出现一个看起来相似但
连字符/下划线不同的用户名并不算成功，必须以该账户实际执行 `id` 为准。

## 4. 最终重启后验收

使用将来真正启动引擎的账户执行：

```bash
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
cat /proc/cmdline

for file in \
  /sys/class/drm/card*/device/mem_info_vram_total \
  /sys/class/drm/card*/device/mem_info_gtt_total; do
  test -r "$file" && printf '%s=' "$file" && cat "$file"
done

id
ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null
/opt/rocm/bin/rocminfo | grep -m1 gfx1151
```

验收值：

- `/proc/cmdline` 同时包含两个内存参数；
- 固定 VRAM 为 `536870912` 字节；
- GTT 为 `103079215104` 字节；
- `id` 包含 `render` 和 `video`；
- `rocminfo` 成功并报告 `gfx1151`。

主机检查通过后再运行：

```bash
./aima-engine doctor
```

## 常见错误状态

| 实际状态 | 含义 | 处理方法 |
|---|---|---|
| `MemTotal` 约 64 GiB、VRAM 64 GiB、GTT 32 GiB | BIOS 使用了 64/64 切分 | 把固定 UMA 改为 512 MiB |
| `MemTotal` 约 125 GiB、VRAM 512 MiB、GTT 62.5 GiB | BIOS 已正确，但缺少 GTT 内核参数 | 添加两个 GRUB 参数并重启 |
| 内存池正确，但 `rocminfo` 无法打开 GPU | 服务账户没有设备权限 | 把准确账户加入 `render,video` 后重新登录 |
| 内存池和权限正确，仍分配失败 | 有其他 GPU/GTT 负载，或运行时/内核不在支持范围 | 停止竞争负载并对齐合格软件环境 |

## 回滚

如需撤销 Linux 内存管理调整，只从 `/etc/default/grub` 删除
`ttm.pages_limit=25165824 amdgpu.gttsize=98304`，执行
`sudo update-grub` 后重启。恢复大块固定显存并不是合格引擎配置，只应在
排查特定固件显示问题时临时使用。
