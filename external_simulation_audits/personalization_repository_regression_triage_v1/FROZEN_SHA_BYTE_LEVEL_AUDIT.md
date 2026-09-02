# Frozen SHA byte-level audit

## 判定

`CROSS_PLATFORM_LINE_ENDING_HASH_INSTABILITY=CONFIRMED`

代表性直接 SHA mismatch 均由 Windows checkout 的 CRLF 与 Git blob 的 LF 造成。没有发现 BOM 或 final-newline 差异；将工作树 CRLF 只在内存中换成 LF 后，6 个文件均与 HEAD Git blob 逐字节相同。本诊断没有改写或规范化磁盘上的证据文件。

## Git 属性与配置

```text
core.autocrlf = true
core.eol = unset
core.safecrlf = unset
.gitattributes = absent
git check-attr text eol = unspecified / unspecified
git ls-files --eol = i/lf w/crlf attr/
```

`core.autocrlf` 来源：`C:/Program Files/Git/etc/gitconfig`。

## 逐字节结果

| file | expected / Git blob SHA-256 | working-tree SHA-256 | blob bytes | work bytes | CRLF count | normalized==blob |
|---|---|---|---:|---:|---:|---|
| `external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json` | `de27c80d3ca93cd299c016ccb5d80032a8af417a2d06b91e2a01e5f0b2680f9e` | `1db5037c59abc656dad268867c733eb92794a9ea5962516898a098ca8d342ca8` | 10014 | 10286 | 272 | true |
| `reference_release/reference_measured_asymmetric_closed_slow.csv` | `f63bdea2e0d346d73151eedaac73e887f1028c99a6eb15cfc3bc44cfd088a881` | `56be79233bd45b9a137fc6f8e10d78bedbe0c4701623c1dcca6033b891f7040a` | 284701 | 285103 | 402 | true |
| `reference_release/reference_cycle_closure_audit.csv` | `4e9ea5d6914f7b3959e5f6809616c87f631e7b17362f5ecca6235d15f055fc13` | `7fb9535987b00be954f9face9e85659be0e6acc50c9a0f8ef2b6307f5bc56799` | 4580 | 4585 | 5 | true |
| `lower_limb_sim/formal_artifacts/admissible_personalization_region_v1/admissible_parameter_samples.csv` | `cf71105b040092e56c581b80c97c1dc1852b75ae27ed263d0f1d1a1681ab0118` | `f79f69f32e662d53e89f4c20c638df4284e1e39af78b7cfeda70d0adc7aea863` | 4005619 | 4017170 | 11551 | true |
| `external_simulation/myoleg_supine_rehab_v1/myoleg_supine_right_v1.xml` | `20e46cd3e372fbcbdddaf7ff6dceae0652e5e9f66237ca573f6612ae4a800b7d` | `960cdb3dca5eeda281a69c99e899bbd1571377e8f12d8c729286948ce9792e59` | 117963 | 119335 | 1372 | true |
| `external_simulation/measurement_driven_personalization_data_and_endpoint_design_v1/build_design.py` | `09cf720914d53d6457269f81d4ee23111cd26f346956d414f37d7b67cca9f7ba` | `b9b70ca0ef33b0d568c39629cca642a7d53564552670a240fbd7f327f12e9263` | 50876 | 51600 | 724 | true |

所有 6 项还同时满足：

- Git blob CRLF 数为 0；Git blob LF 数等于工作树 CRLF 数。
- `working_bytes - blob_bytes == working_CRLF_count`。
- 工作树没有 lone LF；所有换行均为 CRLF。
- 两侧均无 UTF-8 BOM。
- 两侧均保留 final newline。
- UTF-8 语义内容没有额外差异。

## Git blob 与分支

| file | HEAD blob ID | Mac parent blob ID | equal |
|---|---|---|---|
| protocol JSON | `889997afe8ba9971d05e6a5c67e13193aebdf8e6` | `889997afe8ba9971d05e6a5c67e13193aebdf8e6` | true |
| active reference CSV | `bbb2d271b2ff3b7e43d2b556b8a386df4aca3437` | `bbb2d271b2ff3b7e43d2b556b8a386df4aca3437` | true |
| closure audit CSV | `26188d4659a5879c9b6afc55218bd16d35842fe2` | `26188d4659a5879c9b6afc55218bd16d35842fe2` | true |
| admissible samples CSV | `53b4a81fac6df34387c7c6c08abe6189a696a374` | `53b4a81fac6df34387c7c6c08abe6189a696a374` | true |
| MyoLeg XML | `6aeb871e09d4bd70d4eee671ba55675871c9b2d3` | `6aeb871e09d4bd70d4eee671ba55675871c9b2d3` | true |
| builder Python | `6d40d7ee6cd2f30747244ef95c7bf98ce50a9507` | `6d40d7ee6cd2f30747244ef95c7bf98ce50a9507` | true |

这证明 HEAD 与 Mac 父提交的仓库对象对这些文件是相同的，但不证明 Mac 机器物理 worktree 的 checkout bytes。当前 Windows worktree 已明确不同于共同的 canonical blob。

## 最小复现命令

```powershell
python -B -m pytest -q --tb=short external_simulation/test_measurement_driven_personalization_data_and_endpoint_design_v1.py::test_protocol_was_frozen_before_endpoint_decision_outputs
git ls-files --eol external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json
git cat-file blob HEAD:external_simulation_audits/measurement_driven_personalization_data_and_endpoint_design_v1/DATA_AND_ENDPOINT_DESIGN_PROTOCOL.json
```

预期测试输出中的两项 SHA 正是上表第一行的 working-tree 与 Git blob SHA。
