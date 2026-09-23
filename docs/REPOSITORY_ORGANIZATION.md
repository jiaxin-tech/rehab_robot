# 仓库整理记录

整理日期：2026-09-23。[返回文档索引](README.md)。

## 整理后的阅读方式

根 README 负责介绍课题、当前状态、快速验证和目录入口。环境操作统一到 [GETTING_STARTED.md](GETTING_STARTED.md)，研究结论统一到 [RESEARCH.md](RESEARCH.md)，原 README 中的机器人命令与实现契约集中到 [ROBOT_OPERATIONS.md](ROBOT_OPERATIONS.md)。[CODE_MAP.md](CODE_MAP.md)连接代码、测试和结果；[ARTIFACTS.md](ARTIFACTS.md)解释文件保留与清理依据。

根目录原有 21 份专题报告迁入 `docs/research/`、`docs/history/`、`docs/visualization/`，完整位置见[文档索引](README.md)。原 `lower_limb_sim/README.md` 的 Stage 1–6 手册归档为 [LOWER_LIMB_STAGES_1_TO_6.md](history/LOWER_LIMB_STAGES_1_TO_6.md)，包入口改为简短导航。历史报告保留全文，仅修正迁移后失效的相对链接；不同指标、候选域、预算和实验阶段的结果没有合并为同一个科学结论。

## 合并、删除与保留

| 处理 | 结果及原因 |
|---|---|
| 合并导航 | 根目录和下肢包长 README 的阅读入口集中到文档索引；三个视频报告共用一个[可视化入口](visualization/README.md)，详细渲染证据仍各自保留。 |
| 清除重复维护 | 安装命令、研究作用域和机器人操作细节各有一个当前维护位置；根 README 只保留摘要和最短验证命令。 |
| 删除 | 根 `.DS_Store`，12,292 字节的 macOS 显示元数据；可从整理前 Git 历史恢复，不含实验数据。 |
| 保留固定路径 | `bone_return_3_leg.csv`、`CURRENT_ARCHITECTURE.md`、`REAL_ROBOT_EXPERIMENT.md`、冻结参考及实验产物有读取、来源或 SHA 依赖。 |
| 保留完整证据包 | 诊断 CSV/JSON/PNG、同内容的源参考与发布参考、厂商 SDK 原始发行包都有独立职责，不能只按重复字节删除。 |
| 保留用户配置与环境 | `lower_limb_sim/rehab_robot.code-workspace` 引用相邻论文仓库；`.venv/`、`.tools/` 是已验证环境，继续本地忽略。 |

Python 包、import 路径、CLI 名称、配置值、SDK 代码、实验 CSV/NPZ 和冻结校验值在本次结构整理中保持不变。旧标量、P2、信任和诊断算法是有调用或实验对照的历史研究分支，已在代码地图标注，未凭版本号删除。

## 历史复现边界

`external_simulation/static_strap_pull_geometry_validation_protocol_v1/build_protocol.py` 对根 `CURRENT_ARCHITECTURE.md`、`REAL_ROBOT_EXPERIMENT.md` 和 `README.md` 有原路径及精确 SHA 校验。前两份原文保持原位和原字节。根 README 在本次整理前的环境/作用域更新中已经变化，因此该历史 builder 对 README 的旧 SHA 校验需要对应历史版本，不能把当前核心回归通过理解为这个历史协议可以无条件重建。本次没有修改其预期 SHA 或历史 JSON 来掩盖差异。

同样，上一轮 Windows `resource` 兼容修复改变了历史 replay builder 源码；其历史源码校验边界见[上手指南](GETTING_STARTED.md)。生成型报告仍保留在各自实验目录，历史正文中的反引号路径通常是仓库根目录路径或原阶段记录。

## 验证与提交

环境和核心回归配置先提交为 `35fb3fc`（`build: configure Windows Python environment and core regression`），作为本次文档整理的独立基线。结构整理另作提交，便于单独审阅或回退。

整理验收包括当前核心回归、本次维护的 Markdown 相对链接、迁移正文差异、两份固定路径文档的字节一致性以及 Git 差异检查。具体测试环境和实测结果统一记录在[上手指南](GETTING_STARTED.md)；不把本次验收称为历史全仓回归或真机验证。

迁移核对结果：21 份根报告中 20 份内容完全一致，五腿报告仅修正 3 个图片链接；下肢旧手册仅修正 6 个审计链接。当前维护范围内的 32 份 Markdown、276 个本地链接全部可解析到现有文件或目录。暂存差异不包含 Python 源码、配置、实验数据或 SDK 文件变更。

## 后续文件放置约定

- 新的当前操作说明放在 `docs/`，在文档索引增加入口；根 README 保持简短。
- 专题设计或阶段报告放在 `docs/research/`；已被后续版本替代的说明放在 `docs/history/`，保留原阶段和指标定义。
- 生成器输出仍进入自身实验目录，报告与其 CSV/JSON/NPZ 放在一起，由研究指南链接；不要把旧结果覆盖成“最新”。
- 新脚本和算法沿现有模块职责放置，并在代码地图标明入口、输入、输出及对应测试。
- 整理冻结文件前先检查读取路径、manifest 与原始字节 SHA。迁移导航不会自动授权改写实验结论或数据。
