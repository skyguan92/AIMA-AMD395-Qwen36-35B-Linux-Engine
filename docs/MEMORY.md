# AMD395 unified-memory setup

The qualified engine does not use a large fixed BIOS framebuffer as its model
pool. It uses a 96 GiB AMDGPU GTT pool backed by a 128 GB Ryzen AI Max+ 395
system. Both the BIOS UMA split and the Linux kernel parameters must be set
correctly.

## Required post-boot state

| Item | Qualified value |
|---|---:|
| Installed unified memory | 128 GB |
| Linux `MemTotal` | approximately 125 GiB |
| Fixed/visible VRAM | 512 MiB (`536870912` bytes) |
| AMDGPU GTT pool | 96 GiB (`103079215104` bytes) |
| TTM page limit | `25165824` 4 KiB pages |

The full model process reached approximately 73.2 GB allocated memory in the
accepted runs, and maximum-context cases peaked higher. A 64 GB machine is not
large enough for this BF16 release. A 128 GB machine split into 64 GB system
memory plus 64 GB fixed VRAM is also not a valid substitute: that setup
typically exposes only about 32 GiB of GTT.

## 1. Change the BIOS UMA split

This step requires a local console or other reliable out-of-band access.
Firmware names vary; common labels are `UMA Frame Buffer Size`, `iGPU Memory`,
or `Dedicated Graphics Memory`.

1. Enter firmware setup.
2. Set the fixed UMA/framebuffer allocation to **512 MiB**. If 512 MiB is not
   listed, use the smallest setting or `Auto` only when the post-boot check
   below reports exactly 512 MiB.
3. Save and reboot.

Do not leave the firmware at a 64 GB fixed-VRAM split. The model uses the GTT
pool configured in the next step, while a large fixed framebuffer removes
memory from Linux.

After reboot, check the system and fixed-VRAM pools:

```bash
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo

for file in /sys/class/drm/card*/device/mem_info_vram_total; do
  test -r "$file" && printf '%s=' "$file" && cat "$file"
done
```

Expected: `MemTotal` is approximately 125 GiB and
`mem_info_vram_total=536870912`. The DRM card number is not stable across
machines or boots, so use the wildcard rather than assuming `card0`.

## 2. Configure a 96 GiB GTT pool

On the qualified Ubuntu system, add these parameters to the existing GRUB
kernel command line:

```text
ttm.pages_limit=25165824 amdgpu.gttsize=98304
```

`98304` is 96 GiB expressed in MiB. `25165824` is the same capacity expressed
as 4 KiB pages.

Edit `/etc/default/grub` with `sudoedit` and preserve every existing parameter.
For example:

```bash
sudoedit /etc/default/grub
```

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=25165824 amdgpu.gttsize=98304"
```

Then rebuild the boot menu and reboot:

```bash
sudo update-grub
sudo reboot
```

These are global AMDGPU memory-manager settings. Do not apply them to a host
with less than 128 GB of installed unified memory, and do not claim a different
kernel as qualified merely because the memory checks pass. v1.0.0 performance
was qualified on the kernel and ROCm versions listed in
[`INSTALL.md`](INSTALL.md).

## 3. Grant GPU device access

The serving account must be a member of both `render` and `video`:

```bash
sudo usermod -aG render,video "$(id -un)"
```

Log out completely and log in again, or reboot. Merely listing a similar
hyphenated or underscored account name in `/etc/group` is not sufficient; the
actual output of `id` must contain the groups.

## 4. Validate after the final reboot

Run the following as the same account that will start the engine:

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

The acceptance values are:

- `/proc/cmdline` contains both memory parameters;
- fixed VRAM is `536870912` bytes;
- GTT is `103079215104` bytes;
- `id` contains `render` and `video`;
- `rocminfo` succeeds and reports `gfx1151`.

Only after these host checks pass should you run:

```bash
./aima-engine doctor
```

## Common failure states

| Observed state | Meaning | Resolution |
|---|---|---|
| About 64 GiB `MemTotal`, 64 GiB VRAM, 32 GiB GTT | BIOS is using a 64/64 split | Set fixed UMA to 512 MiB |
| About 125 GiB `MemTotal`, 512 MiB VRAM, 62.5 GiB GTT | BIOS is correct; GTT kernel parameters are missing | Add both GRUB parameters and reboot |
| Correct pools, but `rocminfo` cannot open a GPU | Serving account lacks device access | Add the exact account to `render,video`, then re-login |
| Correct pools and access, but allocation still fails | Another GPU/GTT workload or an unsupported runtime/kernel is in use | Stop competing workloads and match the qualified software envelope |

## Rollback

To undo the Linux memory-manager change, remove only
`ttm.pages_limit=25165824 amdgpu.gttsize=98304` from `/etc/default/grub`, run
`sudo update-grub`, and reboot. Restoring a large fixed-VRAM BIOS split is not a
valid engine configuration, but it may be used temporarily when diagnosing
firmware-specific display problems.
