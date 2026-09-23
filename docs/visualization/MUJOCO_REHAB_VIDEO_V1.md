# MuJoCo Rehab Video V1

用于简历、技术面试与导师汇报的固定 ROM 髋膝协调展示。状态：COMPLETE。

仅展示 offline MuJoCo simulation / exact-state replay；复用既有 inverse-dynamics replay 做数值校验。不代表闭环控制、个体最优、舒适性改善或临床有效性。

## 模型与轨迹

使用 `lower_limb_sim/five_leg_mujoco_v1` 冻结模型 `LEG_0_NOMINAL` 及其 `make_rom_profile()`：髋 29–112°，膝 18.5–119.5°。直接选择 `build_leg_domain()` 的 625 点 V3 域，Reference `(0,0)`、Contrast A `(-0.03,+0.03)`、Contrast B `(+0.03,-0.03)`。

三者同 ROM、同 24 秒原始时间数组、同初末 q/dq/ddq。直接使用原采样点，未插值、重拟合或放大运动。保持 `theta_shank=q_hip-q_knee`；MuJoCo knee qpos 为项目 knee 的负值。画面 cuff 标记为等效牵引点，并非脚踝。

## 画面与文件

固定透视侧视：azimuth=90°，elevation=0°，distance=1.5 m，lookat=(0.30,0,0.19) m，原模型默认垂直 FOV=45°。1920×1080，30 fps，MuJoCo offscreen RGB → imageio-ffmpeg/libx264，H.264/yuv420p，faststart。

- `outputs/mujoco_v3_resume_showcase.mp4`：19 秒、570 帧。片头 2 秒，三条轨迹各 5 秒，片尾 2 秒。每条原始 24 秒运动压缩至 5 秒；字幕显示实际仿真时间。
- `outputs/mujoco_v3_side_by_side.mp4`：10 秒、300 帧。Reference 与 Contrast A 同采样点、同相机、同步播放；带角度、phase、真实时间和关节曲线游标。青色为 hip、橙色为 knee。
- `outputs/previews/`：两版各自 start/mid/end PNG。
- 同名 JSON：输出尺寸、时长、相机和最大角度误差。

三画面版未生成，以保持图文清晰。颜色、关节球和短 cuff 轨迹只存在于 rendering scene；轨迹不是力方向。唯一 MjModel 展示配置设置为 offscreen framebuffer 尺寸，不改机械参数。

## 验证

`tests/test_mujoco_rehab_video.py`：3 passed，覆盖既有域/三组 beta/非法 beta、同 ROM 与首末状态、确定性帧索引、逐点角度一致性、真实离屏渲染及机械参数不变。

两版实际渲染：max hip error=0 rad，max knee error=0 rad。解码检查尺寸、30 fps、帧数、无全黑帧及运动帧变化。固定相机的预览检查显示 pelvis、thigh、knee、shank、cuff 完整可见；闭合轨迹首末姿态相同是预期行为，字幕时间和 phase 不同。

## 重新生成

在 repository 根目录执行。当前已配置的临时 Python 环境包含 MuJoCo 3.6.0、numpy/scipy、Pillow、imageio-ffmpeg 0.6.0；未改项目依赖文件。macOS 沙箱环境需要允许 CoreGraphics offscreen 上下文。临时环境清理后，可在独立环境安装这些渲染依赖再替换下面的 Python 路径。

```sh
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_mujoco_rehab_video --mode resume-showcase --leg LEG_0_NOMINAL --output outputs/mujoco_v3_resume_showcase.mp4
/private/tmp/rehab_architecture_review_venv/bin/python -m scripts.render_mujoco_rehab_video --mode side-by-side --leg LEG_0_NOMINAL --output outputs/mujoco_v3_side_by_side.mp4
```

单轨迹支持 `--mode single --beta-flex -0.03 --beta-extend 0.03 --output outputs/v3_contrast_a.mp4`；720p 可用 `--width 1280 --height 720`。

新增文件仅为 visualization package、CLI、针对性测试、本说明与输出媒体。未修改核心仿真、V3、ROM、冻结 benchmark、BO 或控制逻辑；未 commit。
