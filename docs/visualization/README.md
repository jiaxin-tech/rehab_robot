# 离线视频入口

三组展示均为仿真或规定状态回放。画面中的轨迹跟随、示意机器人与数值误差不能解释为真实机器人闭环控制、患者收益或临床验证。

| 需要的画面 | CLI 入口与 mode | 完整说明 |
|---|---|---|
| 固定 ROM 的 V3 参考与对照轨迹 | `scripts.render_mujoco_rehab_video`：`resume-showcase`、`side-by-side`、`single` | [MuJoCo V3 展示](MUJOCO_REHAB_VIDEO_V1.md) |
| E2 分支含义、五腿对比、共同与个体 oracle | `scripts.render_mujoco_rehab_video`：`e2-explainer`、`five-leg-e2`、`e2-common-vs-individual`、`e2-landscapes` | [E2 多腿展示](E2_MULTI_LEG_VIDEO_V1.md) |
| MyoLeg 与示意机器人、四次离线 EI 过程 | `scripts.render_myoleg_robot_video`：`system-showcase`、`reference-vs-selected`、`optimization-demo` | [MyoLeg 系统与优化展示](MYOLEG_ROBOT_OPTIMIZATION_VIDEO_V1.md) |

从仓库根目录执行，使用 [安装指南](../GETTING_STARTED.md)配置的 Python。先查看 CLI 参数；下面的示例生成一段 V3 并排对比：

```powershell
.\.venv\Scripts\python.exe -m scripts.render_mujoco_rehab_video --help
.\.venv\Scripts\python.exe -m scripts.render_mujoco_rehab_video --mode side-by-side --leg LEG_0_NOMINAL --output outputs/mujoco_v3_side_by_side.mp4
```

渲染依赖 MuJoCo 离屏上下文、Pillow 和 `imageio-ffmpeg`；无图形上下文的环境仅通过数值测试不能保证视频可生成。MyoLeg 展示还需要原模型引用的 MyoSuite 网格资产，历史报告记录的 macOS 本地路径不代表当前机器已安装。`optimization-demo` 默认复用已记录结果；`--rerun-optimization` 会重新运行对应四次离线试验。

三份原报告完整保留了场景参数、源模型、输出文件、解码/视觉验证和历史复现命令。更新操作环境时以当前安装指南为准，读取已生成视频及 sidecar JSON 时以各报告列出的产物位置为准。返回 [全部文档](../README.md)或[当前研究](../RESEARCH.md)。
