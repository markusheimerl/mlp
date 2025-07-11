# mlp
A multilayer perceptron implementation

Consider a standard feed-forward network operating on batched inputs of shape (batch_size × input_dim). The architecture consists of two linear transformations with an intermediate swish activation, where the forward propagation follows:

$$
\begin{align*}
Z &= XW_1 \\
A &= Z\sigma(Z) \\
Y &= AW_2
\end{align*}
$$

The swish activation $x\sigma(x)$ interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_2} &= A^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial A} &= (\frac{\partial L}{\partial Y})(W_2)^\top \\
\frac{\partial L}{\partial Z} &= \frac{\partial L}{\partial A} \odot [\sigma(Z) + Z\sigma(Z)(1-\sigma(Z))] \\
\frac{\partial L}{\partial W_1} &= X^\top(\frac{\partial L}{\partial Z})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```

## Benchmarks

### NVIDIA Jetson Orin Nano Super Developer Kit

#### CPU
```
R² score for output y0: 0.99999571
R² score for output y1: 0.99998200
R² score for output y2: 0.99999940
R² score for output y3: 0.99994481
...
3 minutes 39 seconds elapsed
```

#### GPU
```
R² score for output y0: 0.99999571
R² score for output y1: 0.99998200
R² score for output y2: 0.99999940
R² score for output y3: 0.99994481
...
10 seconds elapsed
```

<details>
<summary>Full logs</summary>
```bash
markus@jetson:~/mlp$ sudo inxi -v8 -z
System:
  Kernel: 5.15.148-tegra aarch64 bits: 64 compiler: N/A
    parameters: root=PARTUUID=eeb08ace-b41d-4c4d-ae7e-7a2cbb3db575 rw rootwait rootfstype=ext4
    mminit_loglevel=4 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0
    video=efifb:off console=tty0 bl_prof_dataptr=2031616@0x271E10000
    bl_prof_ro_ptr=65536@0x271E00000
  Console: pty pts/5 Distro: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
Machine:
  Type: Other-vm? System: NVIDIA
    product: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super v: N/A
    serial: <filter> Chassis: type: 1 serial: N/A
  Mobo: NVIDIA model: Jetson serial: <filter> UEFI: EDK II v: 36.4.4-gcid-41062509
    date: 06/16/2025
Battery:
  Message: No system battery data found. Is one present?
Memory:
  RAM: total: 7.44 GiB used: 3.04 GiB (40.9%)
  Array-1: capacity: 8 GiB slots: 1 EC: Single-bit ECC max-module-size: 8 GiB note: est.
PCI Slots:
  Message: No ARM data found for this feature.
CPU:
  Info: model: ARMv8 v8l variant: cortex-a78 bits: 64 type: MCP AMP arch: v8l family: 8
    model-id: 0 stepping: 1
  Topology: variant: cpus: 1x cores: 4 variant: cpus: 1x cores: 2 smt: <unsupported> cache:
    L1: 2x 512 KiB (1024 KiB) desc: d-4x64 KiB; i-4x64 KiB L2: 2x 1024 KiB (2 MiB) desc: 4x256 KiB
    L3: 2x 2 MiB (4 MiB) desc: 1x2 MiB
  Speed (MHz): avg: 730 min/max: 115/1728 base/boost: 1728/1728 scaling: driver: tegra194
    governor: schedutil ext-clock: 31 MHz cores: 1: 730 2: 730 3: 730 4: 730 5: 730 6: 730
    bogomips: 375
  Features: aes asimd asimddp asimdhp asimdrdm atomics cpuid crc32 dcpop evtstrm flagm fp fphp
    ilrcpc lrcpc paca pacg pmull sha1 sha2 uscat
  Vulnerabilities:
  Type: gather_data_sampling status: Not affected
  Type: itlb_multihit status: Not affected
  Type: l1tf status: Not affected
  Type: mds status: Not affected
  Type: meltdown status: Not affected
  Type: mmio_stale_data status: Not affected
  Type: retbleed status: Not affected
  Type: spec_rstack_overflow status: Not affected
  Type: spec_store_bypass mitigation: Speculative Store Bypass disabled via prctl
  Type: spectre_v1 mitigation: __user pointer sanitization
  Type: spectre_v2 mitigation: CSV2, BHB
  Type: srbds status: Not affected
  Type: tsx_async_abort status: Not affected
Graphics:
  Device-1: tegra234-display driver: nv_platform v: N/A bus-ID: N/A chip-ID: nvidia:13800000
    class-ID: display
  Device-2: ga10b driver: gk20a v: N/A bus-ID: N/A chip-ID: nvidia:17000000 class-ID: gpu
  Device-3: ga10b driver: gk20a v: N/A bus-ID: N/A chip-ID: nvidia:gpu class-ID: gpu
  Display: server: X.org v: 1.21.1.4 with: Xwayland v: 22.1.1 driver: X: loaded: N/A
    failed: nvidia gpu: nv_platform,gk20a,gk20a note:  X driver n/a tty: 140x28
  Message: GL data unavailable in console for root.
Audio:
  Device-1: tegra186-audio-graph-card driver: tegra_asoc: bus-ID: N/A chip-ID: nvidia:sound
    class-ID: sound
  Sound Server-1: ALSA v: k5.15.148-tegra running: yes
  Sound Server-2: PulseAudio v: 15.99.1 running: no
  Sound Server-3: PipeWire v: 0.3.48 running: no
Network:
  Device-1: Realtek RTL8822CE 802.11ac PCIe Wireless Network Adapter vendor: AzureWave
    driver: rtl88x2ce v: N/A modules: rtl8822ce port: 1000 bus-ID: 0001:01:00.0 chip-ID: 10ec:c822
    class-ID: 0280
  IF: wlP1p1s0 state: up mac: <filter>
  IP v4: <filter> type: dynamic noprefixroute scope: global broadcast: <filter>
  IP v6: <filter> type: temporary dynamic scope: global
  IP v6: <filter> type: dynamic mngtmpaddr noprefixroute scope: global
  IP v6: <filter> type: temporary dynamic scope: global
  IP v6: <filter> type: dynamic mngtmpaddr noprefixroute scope: global
  IP v6: <filter> type: noprefixroute scope: link
  Device-2: Realtek RTL8111/8168/8411 PCI Express Gigabit Ethernet driver: r8168
    v: 8.053.00-NAPI port: 300000 bus-ID: 0008:01:00.0 chip-ID: 10ec:8168 class-ID: 0200
  IF: enP8p1s0 state: down mac: <filter>
  IF-ID-1: can0 state: down mac: N/A
  IF-ID-2: docker0 state: down mac: <filter>
  IP v4: <filter> scope: global broadcast: <filter>
  IF-ID-3: l4tbr0 state: down mac: <filter>
  IF-ID-4: usb0 state: down mac: <filter>
  IF-ID-5: usb1 state: down mac: <filter>
  WAN IP: <filter>
Bluetooth:
  Device-1: IMC Networks Bluetooth Radio type: USB driver: rtk_btusb
    v: 3.1.6fd4e69.20220818-105856 bus-ID: 1-3:3 chip-ID: 13d3:3549 class-ID: e001 serial: <filter>
  Report: hciconfig ID: hci0 rfk-id: 1 state: up address: <filter> bt-v: 3.0 lmp-v: 5.1
    sub-v: cbc9 hci-v: 5.1 rev: 9a8
  Info: acl-mtu: 1021:6 sco-mtu: 255:12 link-policy: rswitch hold sniff park
    link-mode: peripheral accept
Logical:
  Message: No logical block device data found.
RAID:
  Message: No RAID data found.
Drives:
  Local Storage: total: 931.51 GiB used: 74.33 GiB (8.0%)
  ID-1: /dev/nvme0n1 maj-min: 259:0 vendor: Crucial model: CT1000P3PSSD8 size: 931.51 GiB
    block-size: physical: 512 B logical: 512 B speed: 63.2 Gb/s lanes: 4 type: SSD serial: <filter>
    rev: P9CR413 temp: 56 (329 Kelvin) C scheme: GPT
  SMART: yes health: PASSED on: 7d 0h cycles: 7 read-units: 1,640,274 [839 GB]
    written-units: 794,866 [406 GB]
  Message: No optical or floppy data found.
Partition:
  ID-1: / raw-size: 930.06 GiB size: 914.39 GiB (98.31%) used: 74.33 GiB (8.1%) fs: ext4
    block-size: 4096 B dev: /dev/nvme0n1p1 maj-min: 259:1 label: N/A
    uuid: 96859a1f-2e85-4e69-b6dc-a93e5e036fe5
  ID-2: /boot/efi raw-size: 64 MiB size: 63 MiB (98.44%) used: 110 KiB (0.2%) fs: vfat
    block-size: 512 B dev: /dev/nvme0n1p10 maj-min: 259:10 label: N/A uuid: 2ECE-EED7
Swap:
  Kernel: swappiness: 60 (default) cache-pressure: 100 (default)
  ID-1: swap-1 type: file size: 16 GiB used: 414 MiB (2.5%) priority: -2 file: /16GB.swap
Unmounted:
  ID-1: /dev/nvme0n1p11 maj-min: 259:11 size: 80 MiB fs: N/A label: N/A uuid: N/A
  ID-2: /dev/nvme0n1p12 maj-min: 259:12 size: 512 KiB fs: N/A label: N/A uuid: N/A
  ID-3: /dev/nvme0n1p13 maj-min: 259:13 size: 64 MiB fs: N/A label: N/A uuid: N/A
  ID-4: /dev/nvme0n1p14 maj-min: 259:14 size: 400 MiB fs: N/A label: N/A uuid: N/A
  ID-5: /dev/nvme0n1p15 maj-min: 259:15 size: 479.5 MiB fs: N/A label: N/A uuid: N/A
  ID-6: /dev/nvme0n1p2 maj-min: 259:2 size: 128 MiB fs: ext4 label: N/A uuid: N/A
  ID-7: /dev/nvme0n1p3 maj-min: 259:3 size: 768 KiB fs: N/A label: N/A uuid: N/A
  ID-8: /dev/nvme0n1p4 maj-min: 259:4 size: 31.6 MiB fs: N/A label: N/A uuid: N/A
  ID-9: /dev/nvme0n1p5 maj-min: 259:5 size: 128 MiB fs: ext4 label: N/A uuid: N/A
  ID-10: /dev/nvme0n1p6 maj-min: 259:6 size: 768 KiB fs: N/A label: N/A uuid: N/A
  ID-11: /dev/nvme0n1p7 maj-min: 259:7 size: 31.6 MiB fs: N/A label: N/A uuid: N/A
  ID-12: /dev/nvme0n1p8 maj-min: 259:8 size: 80 MiB fs: N/A label: N/A uuid: N/A
  ID-13: /dev/nvme0n1p9 maj-min: 259:9 size: 512 KiB fs: N/A label: N/A uuid: N/A
USB:
  Hub-1: 1-0:1 info: Hi-speed hub with single TT ports: 4 rev: 2.0 speed: 480 Mb/s
    chip-ID: 1d6b:0002 class-ID: 0900
  Hub-2: 1-2:2 info: Realtek 4-Port USB 2.0 Hub ports: 4 rev: 2.1 speed: 480 Mb/s
    chip-ID: 0bda:5489 class-ID: 0900
  Device-1: 1-3:3 info: IMC Networks Bluetooth Radio type: Bluetooth driver: rtk_btusb
    interfaces: 2 rev: 1.0 speed: 12 Mb/s power: 500mA chip-ID: 13d3:3549 class-ID: e001
    serial: <filter>
  Hub-3: 2-0:1 info: Super-speed hub ports: 4 rev: 3.1 speed: 10 Gb/s chip-ID: 1d6b:0003
    class-ID: 0900
  Hub-4: 2-1:2 info: Realtek 4-Port USB 3.0 Hub ports: 4 rev: 3.2 speed: 10 Gb/s
    chip-ID: 0bda:0489 class-ID: 0900
Sensors:
  Message: No sensor data found. Is lm-sensors configured?
Repos:
  Packages: 2272 apt: 2266 lib: 1166 snap: 6
  Active apt repos in: /etc/apt/sources.list
    1: deb http://ports.ubuntu.com/ubuntu-ports/ jammy main restricted
    2: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main restricted
    3: deb http://ports.ubuntu.com/ubuntu-ports/ jammy universe
    4: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates universe
    5: deb http://ports.ubuntu.com/ubuntu-ports/ jammy multiverse
    6: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates multiverse
    7: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-backports main restricted universe multiverse
    8: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security main restricted
    9: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security universe
    10: deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security multiverse
  Active apt repos in: /etc/apt/sources.list.d/archive_uri-http_apt_llvm_org_jammy_-jammy.list
    1: deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
  Active apt repos in: /etc/apt/sources.list.d/docker.list
    1: deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable
  Active apt repos in: /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
    1: deb https://repo.download.nvidia.com/jetson/common r36.4 main
    2: deb https://repo.download.nvidia.com/jetson/t234 r36.4 main
    3: deb https://repo.download.nvidia.com/jetson/ffmpeg r36.4 main
  Active apt repos in: /etc/apt/sources.list.d/vscode.sources
    1: deb [arch=amd64,arm64,armhf] https://packages.microsoft.com/repos/code stable main
Processes:
  CPU top: 5 of 237
  1: cpu: 3.8% command: node pid: 1871 mem: 601.0 MiB (7.8%)
  2: cpu: 1.6% command: node pid: 7956 mem: 637.0 MiB (8.3%)
  3: cpu: 0.5% command: server-main.js started by: node pid: 1834 mem: 109.3 MiB (1.4%)
  4: cpu: 0.4% command: bootstrap-fork started by: node pid: 2035 mem: 51.6 MiB (0.6%)
  5: cpu: 0.3% command: systemd-udevd pid: 305 mem: 1.90 MiB (0.0%)
  Memory top: 5 of 237
  1: mem: 637.0 MiB (8.3%) command: node pid: 7956 cpu: 1.6%
  2: mem: 601.0 MiB (7.8%) command: node pid: 1871 cpu: 3.8%
  3: mem: 109.3 MiB (1.4%) command: server-main.js started by: node pid: 1834 cpu: 0.5%
  4: mem: 51.6 MiB (0.6%) command: bootstrap-fork started by: node pid: 2035 cpu: 0.4%
  5: mem: 28.5 MiB (0.3%) command: serverworkermain started by: node pid: 11788 cpu: 0.0%
Info:
  Processes: 237 Uptime: 1h 24m Init: systemd v: 249 runlevel: 3 target: multi-user.target
  tool: systemctl Compilers: gcc: 11.4.0 alt: 11 clang: 17.0.6 Shell: Sudo (sudo) v: 1.9.9
  default: Bash v: 5.1.16 running-in: pty pts/5 inxi: 3.3.13
markus@jetson:~/mlp$ make clean && make run
rm -f *.out *.csv *.bin
clang -O3 -march=native -ffast-math -Wall -Wextra mlp.c -static -lopenblas -lm -flto -o mlp.out
Epoch [100/10000], Loss: 772.86669922
Epoch [200/10000], Loss: 555.23937988
Epoch [300/10000], Loss: 437.78024292
Epoch [400/10000], Loss: 324.70727539
Epoch [500/10000], Loss: 217.35476685
Epoch [600/10000], Loss: 130.70704651
Epoch [700/10000], Loss: 71.73543549
Epoch [800/10000], Loss: 37.43667603
Epoch [900/10000], Loss: 19.41950226
Epoch [1000/10000], Loss: 10.31933880
Epoch [1100/10000], Loss: 5.63099289
Epoch [1200/10000], Loss: 3.16765022
Epoch [1300/10000], Loss: 1.83305633
Epoch [1400/10000], Loss: 1.09023440
Epoch [1500/10000], Loss: 0.67012149
Epoch [1600/10000], Loss: 0.42450517
Epoch [1700/10000], Loss: 0.27698073
Epoch [1800/10000], Loss: 0.18541515
Epoch [1900/10000], Loss: 0.12731466
Epoch [2000/10000], Loss: 0.08901355
Epoch [2100/10000], Loss: 0.06330334
Epoch [2200/10000], Loss: 0.04558269
Epoch [2300/10000], Loss: 0.04352096
Epoch [2400/10000], Loss: 0.02428667
Epoch [2500/10000], Loss: 0.01804971
Epoch [2600/10000], Loss: 0.01367102
Epoch [2700/10000], Loss: 0.00998159
Epoch [2800/10000], Loss: 0.00754225
Epoch [2900/10000], Loss: 0.01232514
Epoch [3000/10000], Loss: 0.00443779
Epoch [3100/10000], Loss: 0.00366343
Epoch [3200/10000], Loss: 0.00730362
Epoch [3300/10000], Loss: 0.00334914
Epoch [3400/10000], Loss: 0.00349841
Epoch [3500/10000], Loss: 0.00236098
Epoch [3600/10000], Loss: 0.00146910
Epoch [3700/10000], Loss: 0.00212139
Epoch [3800/10000], Loss: 0.00185858
Epoch [3900/10000], Loss: 0.00318118
Epoch [4000/10000], Loss: 0.00507029
Epoch [4100/10000], Loss: 0.00232823
Epoch [4200/10000], Loss: 0.00074042
Epoch [4300/10000], Loss: 0.00085096
Epoch [4400/10000], Loss: 0.00143243
Epoch [4500/10000], Loss: 0.00166682
Epoch [4600/10000], Loss: 0.00111230
Epoch [4700/10000], Loss: 0.00064408
Epoch [4800/10000], Loss: 0.00155930
Epoch [4900/10000], Loss: 0.00349959
Epoch [5000/10000], Loss: 0.00102399
Epoch [5100/10000], Loss: 0.00692002
Epoch [5200/10000], Loss: 0.00200823
Epoch [5300/10000], Loss: 0.00537116
Epoch [5400/10000], Loss: 0.00118058
Epoch [5500/10000], Loss: 0.00058291
Epoch [5600/10000], Loss: 0.00412479
Epoch [5700/10000], Loss: 0.00208338
Epoch [5800/10000], Loss: 0.00432454
Epoch [5900/10000], Loss: 0.00324754
Epoch [6000/10000], Loss: 0.00321556
Epoch [6100/10000], Loss: 0.00079035
Epoch [6200/10000], Loss: 0.01086480
Epoch [6300/10000], Loss: 0.00100937
Epoch [6400/10000], Loss: 0.00224180
Epoch [6500/10000], Loss: 0.00976298
Epoch [6600/10000], Loss: 0.00319108
Epoch [6700/10000], Loss: 0.00213097
Epoch [6800/10000], Loss: 0.00101675
Epoch [6900/10000], Loss: 0.00314518
Epoch [7000/10000], Loss: 0.00257870
Epoch [7100/10000], Loss: 0.00161028
Epoch [7200/10000], Loss: 0.00209175
Epoch [7300/10000], Loss: 0.00362230
Epoch [7400/10000], Loss: 0.00349579
Epoch [7500/10000], Loss: 0.00990947
Epoch [7600/10000], Loss: 0.00120820
Epoch [7700/10000], Loss: 0.00659988
Epoch [7800/10000], Loss: 0.00085033
Epoch [7900/10000], Loss: 0.00151925
Epoch [8000/10000], Loss: 0.01238815
Epoch [8100/10000], Loss: 0.00185664
Epoch [8200/10000], Loss: 0.00100777
Epoch [8300/10000], Loss: 0.00052285
Epoch [8400/10000], Loss: 0.00080584
Epoch [8500/10000], Loss: 0.00935966
Epoch [8600/10000], Loss: 0.00425243
Epoch [8700/10000], Loss: 0.00127497
Epoch [8800/10000], Loss: 0.00227260
Epoch [8900/10000], Loss: 0.01787902
Epoch [9000/10000], Loss: 0.02244452
Epoch [9100/10000], Loss: 0.00083932
Epoch [9200/10000], Loss: 0.00177002
Epoch [9300/10000], Loss: 0.00067845
Epoch [9400/10000], Loss: 0.00043920
Epoch [9500/10000], Loss: 0.00341275
Epoch [9600/10000], Loss: 0.00722067
Epoch [9700/10000], Loss: 0.01084289
Epoch [9800/10000], Loss: 0.00251675
Epoch [9900/10000], Loss: 0.00731424
Epoch [10000/10000], Loss: 0.00354190
Model saved to 20250711_103434_model.bin
Data saved to 20250711_103434_data.csv

Verifying saved model...
Model loaded from 20250711_103434_model.bin
Loss with loaded model: 0.00354190

Evaluating model performance...

R² scores:
R² score for output y0: 0.99993593
R² score for output y1: 0.99999607
R² score for output y2: 0.99999917
R² score for output y3: 0.99929208

Sample Predictions (first 15 samples):
Output          Predicted       Actual          Difference
------------------------------------------------------------

y0:
Sample 0:          0.729           0.801          -0.072
Sample 1:          0.478           0.502          -0.024
Sample 2:          1.395           1.454          -0.059
Sample 3:        -10.304         -10.250          -0.054
Sample 4:          0.360           0.423          -0.063
Sample 5:          2.230           2.277          -0.047
Sample 6:          5.810           5.866          -0.057
Sample 7:          3.648           3.677          -0.029
Sample 8:         15.364          15.441          -0.077
Sample 9:          1.117           1.166          -0.049
Sample 10:       -10.349         -10.287          -0.062
Sample 11:        -4.473          -4.423          -0.050
Sample 12:        23.031          23.110          -0.079
Sample 13:         4.957           5.034          -0.077
Sample 14:         2.516           2.569          -0.053
Mean Absolute Error for y0: 0.060

y1:
Sample 0:         14.357          14.362          -0.005
Sample 1:          2.321           2.334          -0.013
Sample 2:         -6.857          -6.813          -0.044
Sample 3:         -0.452          -0.443          -0.008
Sample 4:        -18.701         -18.683          -0.018
Sample 5:          2.842           2.859          -0.017
Sample 6:          6.117           6.145          -0.028
Sample 7:         -5.161          -5.151          -0.010
Sample 8:        -14.357         -14.297          -0.060
Sample 9:         -3.171          -3.117          -0.054
Sample 10:         5.313           5.338          -0.026
Sample 11:        25.580          25.592          -0.012
Sample 12:       -13.069         -13.036          -0.033
Sample 13:        17.392          17.433          -0.041
Sample 14:        14.370          14.408          -0.038
Mean Absolute Error for y1: 0.031

y2:
Sample 0:          1.737           1.741          -0.004
Sample 1:          1.143           1.106           0.037
Sample 2:          0.261           0.336          -0.075
Sample 3:          0.869           0.982          -0.114
Sample 4:         -2.019          -2.003          -0.016
Sample 5:          5.108           5.093           0.016
Sample 6:          1.317           1.283           0.034
Sample 7:          2.787           2.837          -0.050
Sample 8:        158.567         158.620          -0.052
Sample 9:       -185.948        -185.927          -0.021
Sample 10:        -0.816          -0.797          -0.019
Sample 11:        -4.194          -4.147          -0.046
Sample 12:         3.800           3.762           0.038
Sample 13:       -19.501         -19.468          -0.033
Sample 14:        -0.800          -0.738          -0.062
Mean Absolute Error for y2: 0.042

y3:
Sample 0:         -0.859          -0.948           0.089
Sample 1:          1.632           1.561           0.071
Sample 2:          2.266           2.223           0.043
Sample 3:         -0.300          -0.397           0.097
Sample 4:          0.495           0.417           0.078
Sample 5:          7.616           7.510           0.106
Sample 6:          2.806           2.757           0.049
Sample 7:         -1.073          -1.152           0.079
Sample 8:          0.294           0.204           0.090
Sample 9:         -0.584          -0.643           0.059
Sample 10:         0.383           0.289           0.094
Sample 11:        -0.149          -0.250           0.101
Sample 12:        -1.361          -1.434           0.073
Sample 13:         1.007           0.922           0.086
Sample 14:         1.452           1.380           0.072
Mean Absolute Error for y3: 0.074
405.76user 470.34system 3:39.45elapsed 399%CPU (0avgtext+0avgdata 25564maxresident)k
0inputs+0outputs (1major+6705minor)pagefaults 0swaps
markus@jetson:~/mlp$ cd gpu
markus@jetson:~/mlp/gpu$ make clean && make run
rm -f *.out *.csv *.bin
clang -O3 -march=native -ffast-math -Wall -Wextra --cuda-gpu-arch=sm_87 -x cuda -Wno-unknown-cuda-version mlp.c -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -flto -o mlp.out
Epoch [100/10000], Loss: 525.70886230
Epoch [200/10000], Loss: 390.09210205
Epoch [300/10000], Loss: 304.37673950
Epoch [400/10000], Loss: 212.64469910
Epoch [500/10000], Loss: 131.69268799
Epoch [600/10000], Loss: 72.43382263
Epoch [700/10000], Loss: 36.01221848
Epoch [800/10000], Loss: 17.02735329
Epoch [900/10000], Loss: 8.18878937
Epoch [1000/10000], Loss: 4.15639830
Epoch [1100/10000], Loss: 2.23800850
Epoch [1200/10000], Loss: 1.25467181
Epoch [1300/10000], Loss: 0.73036027
Epoch [1400/10000], Loss: 0.44016737
Epoch [1500/10000], Loss: 0.27353841
Epoch [1600/10000], Loss: 0.17463812
Epoch [1700/10000], Loss: 0.11387262
Epoch [1800/10000], Loss: 0.07580589
Epoch [1900/10000], Loss: 0.05312770
Epoch [2000/10000], Loss: 0.03933780
Epoch [2100/10000], Loss: 0.02381817
Epoch [2200/10000], Loss: 0.01723091
Epoch [2300/10000], Loss: 0.01182945
Epoch [2400/10000], Loss: 0.00809875
Epoch [2500/10000], Loss: 0.00637777
Epoch [2600/10000], Loss: 0.00607561
Epoch [2700/10000], Loss: 0.01315711
Epoch [2800/10000], Loss: 0.00308387
Epoch [2900/10000], Loss: 0.00245503
Epoch [3000/10000], Loss: 0.00256538
Epoch [3100/10000], Loss: 0.00154548
Epoch [3200/10000], Loss: 0.00298179
Epoch [3300/10000], Loss: 0.00227714
Epoch [3400/10000], Loss: 0.00159975
Epoch [3500/10000], Loss: 0.00164942
Epoch [3600/10000], Loss: 0.01028583
Epoch [3700/10000], Loss: 0.01803835
Epoch [3800/10000], Loss: 0.01597488
Epoch [3900/10000], Loss: 0.00286823
Epoch [4000/10000], Loss: 0.00101559
Epoch [4100/10000], Loss: 0.00078156
Epoch [4200/10000], Loss: 0.00439224
Epoch [4300/10000], Loss: 0.00414590
Epoch [4400/10000], Loss: 0.01668384
Epoch [4500/10000], Loss: 0.00139243
Epoch [4600/10000], Loss: 0.00086021
Epoch [4700/10000], Loss: 0.01313980
Epoch [4800/10000], Loss: 0.01483412
Epoch [4900/10000], Loss: 0.01629071
Epoch [5000/10000], Loss: 0.00388110
Epoch [5100/10000], Loss: 0.01502189
Epoch [5200/10000], Loss: 0.00339311
Epoch [5300/10000], Loss: 0.00145811
Epoch [5400/10000], Loss: 0.00427303
Epoch [5500/10000], Loss: 0.00279603
Epoch [5600/10000], Loss: 0.00181482
Epoch [5700/10000], Loss: 0.00105604
Epoch [5800/10000], Loss: 0.00318544
Epoch [5900/10000], Loss: 0.00596122
Epoch [6000/10000], Loss: 0.00120202
Epoch [6100/10000], Loss: 0.00091760
Epoch [6200/10000], Loss: 0.03663751
Epoch [6300/10000], Loss: 0.00555184
Epoch [6400/10000], Loss: 0.00178462
Epoch [6500/10000], Loss: 0.00476838
Epoch [6600/10000], Loss: 0.00074216
Epoch [6700/10000], Loss: 0.00144520
Epoch [6800/10000], Loss: 0.00068866
Epoch [6900/10000], Loss: 0.00452703
Epoch [7000/10000], Loss: 0.00326122
Epoch [7100/10000], Loss: 0.00565284
Epoch [7200/10000], Loss: 0.00430115
Epoch [7300/10000], Loss: 0.00350928
Epoch [7400/10000], Loss: 0.00260725
Epoch [7500/10000], Loss: 0.00095790
Epoch [7600/10000], Loss: 0.00117672
Epoch [7700/10000], Loss: 0.00166563
Epoch [7800/10000], Loss: 0.01858453
Epoch [7900/10000], Loss: 0.00140039
Epoch [8000/10000], Loss: 0.00100592
Epoch [8100/10000], Loss: 0.01275022
Epoch [8200/10000], Loss: 0.00701921
Epoch [8300/10000], Loss: 0.00261399
Epoch [8400/10000], Loss: 0.00182548
Epoch [8500/10000], Loss: 0.00656659
Epoch [8600/10000], Loss: 0.00253848
Epoch [8700/10000], Loss: 0.00077550
Epoch [8800/10000], Loss: 0.00078650
Epoch [8900/10000], Loss: 0.00602606
Epoch [9000/10000], Loss: 0.00306344
Epoch [9100/10000], Loss: 0.00045925
Epoch [9200/10000], Loss: 0.00115071
Epoch [9300/10000], Loss: 0.00284358
Epoch [9400/10000], Loss: 0.00047379
Epoch [9500/10000], Loss: 0.00458850
Epoch [9600/10000], Loss: 0.00462586
Epoch [9700/10000], Loss: 0.00047372
Epoch [9800/10000], Loss: 0.00836545
Epoch [9900/10000], Loss: 0.00262428
Epoch [10000/10000], Loss: 0.00200227
Model saved to 20250711_103555_model.bin
Data saved to 20250711_103555_data.csv

Verifying saved model...
Model loaded from 20250711_103555_model.bin
Loss with loaded model: 0.00200227

Evaluating model performance...

R² scores:
R² score for output y0: 0.99999571
R² score for output y1: 0.99998200
R² score for output y2: 0.99999940
R² score for output y3: 0.99994481

Sample Predictions (first 15 samples):
Output          Predicted       Actual          Difference
------------------------------------------------------------

y0:
Sample 0:          3.079           3.069           0.010
Sample 1:          3.507           3.502           0.005
Sample 2:          3.786           3.799          -0.013
Sample 3:         -3.527          -3.523          -0.003
Sample 4:          8.614           8.629          -0.015
Sample 5:          4.169           4.177          -0.008
Sample 6:         -0.471          -0.473           0.002
Sample 7:         14.186          14.192          -0.006
Sample 8:          2.690           2.694          -0.004
Sample 9:          4.003           4.014          -0.011
Sample 10:         8.800           8.785           0.015
Sample 11:         5.613           5.613           0.001
Sample 12:         0.561           0.559           0.002
Sample 13:         2.908           2.914          -0.006
Sample 14:       -12.307         -12.305          -0.001
Mean Absolute Error for y0: 0.012

y1:
Sample 0:          2.848           2.754           0.093
Sample 1:         34.615          34.497           0.118
Sample 2:          1.561           1.489           0.072
Sample 3:          4.933           4.869           0.064
Sample 4:          2.837           2.756           0.081
Sample 5:         43.861          43.773           0.088
Sample 6:         48.551          48.475           0.076
Sample 7:          8.743           8.673           0.070
Sample 8:         20.493          20.428           0.065
Sample 9:          7.600           7.507           0.094
Sample 10:         1.080           0.991           0.089
Sample 11:        13.048          12.976           0.073
Sample 12:         2.941           2.848           0.093
Sample 13:        15.018          14.933           0.085
Sample 14:        35.711          35.626           0.085
Mean Absolute Error for y1: 0.074

y2:
Sample 0:          3.105           3.118          -0.013
Sample 1:          1.433           1.487          -0.054
Sample 2:          0.257           0.285          -0.028
Sample 3:          1.058           1.072          -0.014
Sample 4:          1.103           1.111          -0.008
Sample 5:         -1.843          -1.791          -0.052
Sample 6:         -0.488          -0.424          -0.064
Sample 7:          1.128           1.130          -0.002
Sample 8:          0.303           0.303           0.000
Sample 9:         19.411          19.401           0.010
Sample 10:         1.348           1.331           0.017
Sample 11:         0.978           0.981          -0.004
Sample 12:        -1.298          -1.318           0.020
Sample 13:         3.955           3.910           0.044
Sample 14:         0.108           0.126          -0.019
Mean Absolute Error for y2: 0.028

y3:
Sample 0:          0.992           1.013          -0.021
Sample 1:          3.306           3.324          -0.018
Sample 2:          2.347           2.377          -0.030
Sample 3:          0.562           0.558           0.004
Sample 4:          3.732           3.759          -0.028
Sample 5:          0.588           0.610          -0.022
Sample 6:         -0.295          -0.291          -0.004
Sample 7:          6.585           6.611          -0.026
Sample 8:         -0.845          -0.837          -0.008
Sample 9:          1.016           1.054          -0.039
Sample 10:         8.761           8.803          -0.043
Sample 11:        -0.099          -0.082          -0.017
Sample 12:         7.126           7.114           0.012
Sample 13:        -0.322          -0.306          -0.016
Sample 14:         3.741           3.744          -0.003
Mean Absolute Error for y3: 0.018
3.06user 0.90system 0:10.28elapsed 38%CPU (0avgtext+0avgdata 224212maxresident)k
0inputs+0outputs (883major+16049minor)pagefaults 0swaps
markus@jetson:~/mlp/gpu$ 
```
</details>