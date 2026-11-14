# 历史协议输入快照

formal_experiment_manifest.json 是 V3 轨迹参数化研究实际绑定的原始字节，来自 Git 提交 86b9ddd3，SHA-256 为 f80a44e24c1b3ce3ffb98f459364883a9798c179f8b7cfa7e05cc9eb401ad441。

后续提交 585e8064 修改了当前 config/formal_experiment_manifest.json 的现场安全审核字段，使历史生成器对当前文件的原始字节检查失败。历史研究改为读取此快照，原冻结校验值、研究结果和当前现场配置均保持各自身份。

当前任务的 ROM、参考轨迹身份等科学字段仍与快照逐字段核对；现场审核字段不构成这个历史离线试验的结果，也不授予运动权限。
