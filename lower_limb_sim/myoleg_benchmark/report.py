"""Evaluator-only summaries and plots from MyoLeg benchmark result files.

This module imports no simulator or learner and never chooses a recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260923
BOOTSTRAP_REPEATS = 2000
NUMERICAL_TOLERANCE = 1e-12
KEYS = ["subject_id", "family", "noise_std", "seed", "method", "budget"]
SUCCESS_STATUSES = {"OK", "SUCCESS", "COMPLETE", "COMPLETED"}
METRICS = [
    "selection_loss", "true_E3", "true_E2", "true_peak_ratio", "improvement_pct",
    "constraint_violations", "invalid_trials", "executed_trials", "elapsed_s",
    "true_feasible_rate", "failure_rate",
]
GROUP = ["scope", "family", "noise_std", "method", "budget"]
LABELS = {
    "REFERENCE": "Reference", "RANDOM": "Random", "SPACE_FILLING": "3D space filling",
    "PHYSICS_GREEDY": "Physics greedy", "RESIDUAL_GP_GREEDY": "Residual GP mean greedy",
    "PURE_BO_EI": "Pure BO EI", "MODEL_INFORMED_BO_EI": "Model-informed BO EI",
}
DEVELOPMENT_IDS = {f"MYOLEG_VP_{i:03d}" for i in range(1, 33) if i % 4}


def scope_for(subject_id: str) -> str:
    if subject_id in DEVELOPMENT_IDS:
        return "DEVELOPMENT"
    if any(word in subject_id.upper() for word in ("NATIVE", "NOMINAL")):
        return "NATIVE"
    return "OTHER"


def bootstrap_mean(values, *, seed=BOOTSTRAP_SEED, repeats=BOOTSTRAP_REPEATS):
    """Percentile interval for independent units supplied by the caller."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    average = float(values.mean())
    if len(values) < 2:
        return average, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return average, float(lo), float(hi)


def interval_includes_numerical_zero(lo, hi):
    """Treat floating-point differences within 1e-12 as zero, not an effect size."""
    return bool(not np.isfinite([lo, hi]).all()
                or (lo <= NUMERICAL_TOLERANCE and hi >= -NUMERICAL_TOLERANCE))


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_result_identities(protocol):
    """Reconstruct the declared schedule without importing the experiment runner."""
    expected = set()
    for subject, family, method in itertools.product(
        protocol["subjects"], protocol["families"], protocol["methods"],
    ):
        seeds = protocol["seeds"]["random"] if method == "RANDOM" else [0]
        conditions = [(0., seeds)]
        if protocol["noise_subjects"] == "all" or subject == "MYOLEG_NATIVE_P0":
            conditions.extend((noise, protocol["seeds"]["paired_noise"])
                              for noise in protocol["noise_levels"])
        for noise, condition_seeds in conditions:
            for seed, budget in itertools.product(condition_seeds, protocol["budgets"]):
                expected.add((subject, family, noise, seed, method, budget))
    return expected


def validate_completed_results(output, protocol, data):
    completion_path = output / "completion.json"
    if not completion_path.is_file():
        raise ValueError("BENCHMARK_NOT_COMPLETE: completion.json is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("completed") is not True:
        raise ValueError("BENCHMARK_NOT_COMPLETE: completed must be true")
    protocol_sha = _file_sha(output / "protocol.json")
    if completion.get("protocol_sha256") != protocol_sha:
        raise ValueError("COMPLETION_PROTOCOL_SHA_MISMATCH")
    actual = set(data[KEYS].itertuples(index=False, name=None))
    expected = expected_result_identities(protocol)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(
            f"RESULT_IDENTITY_COVERAGE_MISMATCH: missing={len(missing)}, extra={len(extra)}"
        )
    if completion.get("result_rows") != len(data):
        raise ValueError("COMPLETION_RESULT_ROW_COUNT_MISMATCH")
    original_report = [value for key, value in protocol["source_sha256"].items()
                       if key.replace("\\", "/") == "lower_limb_sim/myoleg_benchmark/report.py"]
    if len(original_report) != 1:
        raise ValueError("PROTOCOL_REPORT_SOURCE_IDENTITY_MISSING")
    return dict(
        report_sha256=_file_sha(__file__),
        protocol_report_sha256=original_report[0],
        protocol_sha256=protocol_sha,
        completion_sha256=_file_sha(completion_path),
        results_sha256=_file_sha(output / "results.csv"),
        trial_history_sha256=(_file_sha(output / "trial_history.csv")
                              if (output / "trial_history.csv").is_file() else None),
        numerical_tolerance=NUMERICAL_TOLERANCE,
        change_scope=("Report-only correction: 1e-12 numerical tie tolerance, completed-run "
                      "and exact schedule coverage validation, and primary/secondary budget labels. "
                      "No simulation, recommendation, protocol, or result records are changed. "
                      "The tolerance is numerical, not a minimum meaningful effect threshold."),
    )


def prepare_results(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["status", "recommendation_id", "true_feasible"] + METRICS[:-2])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    if frame.empty:
        raise ValueError("No benchmark result rows")
    data = frame.copy()
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate result identity")
    for column in ["noise_std", "seed", "budget"] + METRICS[:-2]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data["selection_loss"]).all():
        raise ValueError("Every result, including a failure, requires finite selection_loss")
    bool_values = data["true_feasible"].astype(str).str.strip().str.lower()
    if not bool_values.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("true_feasible must contain explicit boolean values")
    feasible = bool_values.isin(["true", "1", "1.0"])
    successful = data["status"].astype(str).str.upper().isin(SUCCESS_STATUSES)
    has_recommendation = data["recommendation_id"].notna() & data["recommendation_id"].astype(str).str.strip().ne("")
    if (feasible & (~has_recommendation | ~np.isfinite(data["true_E3"]))).any():
        raise ValueError("A feasible recommendation needs an identity and finite true_E3")
    expected_loss = np.where(
        successful & has_recommendation & np.isfinite(data["true_E3"]),
        np.where(feasible, data["true_E3"], np.maximum(1., data["true_E3"])), 1.,
    )
    if not np.allclose(data["selection_loss"], expected_loss, atol=1e-10, rtol=1e-9):
        raise ValueError("selection_loss disagrees with the frozen recommendation scoring rule")
    data["true_feasible_rate"] = feasible.astype(float)
    data["failure_rate"] = (~successful).astype(float)
    data["scope"] = data["subject_id"].astype(str).map(scope_for)
    return data


def summarize(data: pd.DataFrame):
    """Average seeds within subject first, then give subjects equal weight."""
    subject_group = GROUP + ["subject_id"]
    per_subject = data.groupby(subject_group, sort=True, dropna=False)[METRICS].mean().reset_index()
    sizes = data.groupby(subject_group, sort=True, dropna=False).agg(
        seed_count=("seed", "nunique"), runs=("seed", "size"), failure_runs=("failure_rate", "sum"),
    ).reset_index()
    per_subject = per_subject.merge(sizes, on=subject_group, validate="one_to_one")
    summary = per_subject.groupby(GROUP, sort=True, dropna=False)[METRICS].mean().reset_index()
    counts = per_subject.groupby(GROUP, sort=True, dropna=False).agg(
        subject_count=("subject_id", "nunique"), runs=("runs", "sum"),
        failure_runs=("failure_runs", "sum"), seeds_min=("seed_count", "min"),
        seeds_max=("seed_count", "max"),
    ).reset_index()
    summary = summary.merge(counts, on=GROUP, validate="one_to_one")
    return per_subject, summary


def pairwise_comparisons(per_subject: pd.DataFrame) -> pd.DataFrame:
    columns = ["family", "noise_std", "budget", "method_a", "method_b", "paired_subjects",
               "metric", "mean_a_minus_b", "ci95_low", "ci95_high", "ci_includes_zero"]
    rows = []
    development = per_subject[(per_subject.scope == "DEVELOPMENT") & per_subject.budget.isin([4, 8])]
    for (family, noise, budget), group in development.groupby(["family", "noise_std", "budget"]):
        for a, b in itertools.combinations(sorted(group.method.unique()), 2):
            left = group[group.method == a].set_index("subject_id")
            right = group[group.method == b].set_index("subject_id")
            shared = sorted(set(left.index) & set(right.index))
            for metric in ("selection_loss", "true_feasible_rate"):
                differences = left.loc[shared, metric].to_numpy() - right.loc[shared, metric].to_numpy()
                mean, lo, hi = bootstrap_mean(differences)
                rows.append(dict(family=family, noise_std=noise, budget=int(budget),
                                 method_a=a, method_b=b, paired_subjects=len(shared), metric=metric,
                                 mean_a_minus_b=mean, ci95_low=lo, ci95_high=hi,
                                 ci_includes_zero=interval_includes_numerical_zero(lo, hi)))
    return pd.DataFrame(rows, columns=columns)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "本阶段暂无对应结果。"
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in values) + " |" for values in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def _winner_text(per_subject, summary, protocol):
    messages = []
    expected_methods = set(protocol.get("methods", []))
    subset = summary[(summary.scope == "DEVELOPMENT") & (summary.noise_std == 0) & summary.budget.isin([4, 8])]
    for (family, budget), group in subset.groupby(["family", "budget"]):
        ordered = group.sort_values(["true_feasible_rate", "selection_loss", "method"], ascending=[False, True, True])
        leader = ordered.iloc[0]
        details = per_subject[(per_subject.scope == "DEVELOPMENT") & (per_subject.family == family)
                              & (per_subject.noise_std == 0) & (per_subject.budget == budget)]
        complete = all(set(part.subject_id) == DEVELOPMENT_IDS for _, part in details.groupby("method"))
        complete = complete and len(group) > 1 and (not expected_methods or set(group.method) == expected_methods)
        seed_settings = protocol.get("seeds", {})
        if isinstance(seed_settings, dict) and "random" in seed_settings:
            for method, part in details.groupby("method"):
                expected_count = len(seed_settings["random"]) if method == "RANDOM" else 1
                complete = complete and bool((part.seed_count == expected_count).all())
        budget_label = "主预算" if budget == protocol.get("primary_budget", 4) else "次级预算"
        prefix = f"- {family}，{budget_label} K={int(budget)}："
        if not complete:
            messages.append(prefix + "24个development主体或预定方法覆盖不完整，暂不宣布队列赢家。")
            continue
        first = details[details.method == leader.method].set_index("subject_id").sort_index()
        ambiguous = []
        for opponent in ordered.method.iloc[1:]:
            second = details[details.method == opponent].set_index("subject_id").reindex(first.index)
            feasible_mean, feasible_lo, _ = bootstrap_mean(first.true_feasible_rate - second.true_feasible_rate)
            _, _, loss_hi = bootstrap_mean(first.selection_loss - second.selection_loss)
            if feasible_mean > NUMERICAL_TOLERANCE:
                supported = feasible_lo > NUMERICAL_TOLERANCE
            else:
                supported = feasible_lo >= -NUMERICAL_TOLERANCE and loss_hi < -NUMERICAL_TOLERANCE
            if not supported:
                ambiguous.append(opponent)
        score = f"经验排序首位为 {leader.method}（真实合格率{leader.true_feasible_rate:.3%}，selection_loss={leader.selection_loss:.8g}）"
        if ambiguous:
            ending = "；与 " + "、".join(ambiguous) + " 的配对区间未支持唯一优势，不能宣称唯一赢家。"
        else:
            ending = "；该开发样本的逐对区间支持此排序，仍不构成held-out确认或经多重比较校正的显著性结论。"
        messages.append(prefix + score + ending)
    return "\n".join(messages) or "尚无可供K=4/8排序的development结果。"


def _plots(data, per_subject, summary, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(data.family.unique())
    methods = sorted(data.method.unique(), key=lambda name: list(LABELS).index(name) if name in LABELS else 99)
    colors = {method: plt.get_cmap("tab10")(index % 10) for index, method in enumerate(methods)}
    scopes = [scope for scope in ("DEVELOPMENT", "NATIVE") if scope in set(data.scope)]
    fig, axes = plt.subplots(max(1, len(scopes)), len(families), figsize=(7 * len(families), 4.3 * max(1, len(scopes))), squeeze=False)
    for row, scope in enumerate(scopes):
        for column, family in enumerate(families):
            ax = axes[row, column]
            for method in methods:
                selected = summary[(summary.scope == scope) & (summary.family == family)
                                   & (summary.noise_std == 0) & (summary.method == method)].sort_values("budget")
                if not selected.empty:
                    ax.plot(selected.budget, selected.selection_loss, marker="o", label=LABELS.get(method, method), color=colors[method], linewidth=1.6)
            ax.set(title=f"{scope} | {family}", xlabel="Total trial budget (reference included)", ylabel="Final recommendation selection loss (lower is better)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Development benchmark | seeds averaged within subject, then subjects equally weighted", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(families), figsize=(7 * len(families), 8.4), squeeze=False)
    for row, budget in enumerate((4, 8)):
        for column, family in enumerate(families):
            ax = axes[row, column]
            native = data[(data.scope == "NATIVE") & (data.family == family) & (data.budget == budget)]
            if native.empty:
                ax.text(.5, .5, "No native results at this budget", transform=ax.transAxes, ha="center")
            elif native.subject_id.nunique() != 1:
                raise ValueError("Native repetition intervals require exactly one native model")
            for method in methods:
                selected = native[native.method == method]
                points = []
                for noise, part in selected.groupby("noise_std"):
                    mean, lo, hi = bootstrap_mean(part.selection_loss)
                    points.append((100 * noise, mean, lo, hi))
                if points:
                    values = np.asarray(points)
                    ax.plot(values[:, 0], values[:, 1], marker="o", label=LABELS.get(method, method), color=colors[method])
                    ax.fill_between(values[:, 0], values[:, 2], values[:, 3], alpha=.13, color=colors[method])
            ax.set(title=f"Native | {family} | K={budget}", xlabel="Sample noise / reference joint RMS (%)", ylabel="Final recommendation selection loss")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Native noise robustness | 95% seed-bootstrap intervals describe simulation repeats, not subjects", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "noise_robustness.png", dpi=180)
    plt.close(fig)


def generate_report(output_dir: str | Path) -> dict:
    output = Path(output_dir)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    data = prepare_results(pd.read_csv(output / "results.csv"))
    analysis_provenance = validate_completed_results(output, protocol, data)
    per_subject, summary = summarize(data)
    pairs = pairwise_comparisons(per_subject)
    per_subject.to_csv(output / "subject_summary.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pairs.to_csv(output / "pairwise_comparisons.csv", index=False)
    _plots(data, per_subject, summary, output)
    development = data[data.scope == "DEVELOPMENT"]
    native = data[data.scope == "NATIVE"]
    final_rows = data.loc[data.groupby(KEYS[:-1], sort=False).budget.idxmax()]
    failure_rows = final_rows[final_rows.failure_rate > 0]
    failures = failure_rows.groupby(["scope", "method", "status"]).size().rename("运行数").reset_index()
    key_columns = ["family", "method", "budget", "subject_count", "true_feasible_rate", "selection_loss", "improvement_pct", "constraint_violations", "invalid_trials", "failure_runs"]
    primary = summary[(summary.scope == "DEVELOPMENT") & (summary.noise_std == 0) & summary.budget.isin([4, 8])][key_columns]
    native_table = summary[(summary.scope == "NATIVE") & summary.budget.isin([4, 8])][["noise_std", "seeds_min", "seeds_max"] + key_columns]
    paired_main = pairs[(pairs.noise_std == 0) & (pairs.metric == "selection_loss")
                        & ((pairs.method_a == "MODEL_INFORMED_BO_EI") | (pairs.method_b == "MODEL_INFORMED_BO_EI"))]
    paired_main = paired_main.drop(columns=["noise_std", "metric"])
    config = {key: protocol[key] for key in ("experiment_id", "stage_id", "methods", "subjects", "families", "noise_levels", "noise_subjects", "seeds", "budgets", "primary_budget", "primary_noise_std", "constraints", "current_code", "git", "git_commit", "provenance") if key in protocol}
    unknown = sorted(set(data[data.scope == "OTHER"].subject_id))
    text = f"""# MyoLeg 开发阶段算法实验报告

本报告仅分析已完成的 `results.csv` 与 `protocol.json`，不运行模型、不选推荐、不读取oracle或held-out真值。已有{len(data)}条预算记录，对应{len(final_rows)}个方法/种子顺序运行；覆盖{development.subject_id.nunique()}/24个development主体及{native.subject_id.nunique()}个native模型。Native不计入development队列均值。未知主体分类：{unknown or '无'}。

## 主结果与经验选优

{_winner_text(per_subject, summary, protocol)}

主结论固定为无噪声、总预算 K=4；K=8 是次级预算比较，不替换主结果。排序先看最终推荐真实合格率，再看selection_loss；该词典序来自运行前已固定并记录源码SHA的报告实现。

上述区间是主体配对的描述性bootstrap；与数值零区间 ±{NUMERICAL_TOLERANCE:g} 相交时不宣称唯一优势。该容差仅处理浮点舍入，不是实际效果阈值，也不代表达到它便具有实际价值。本轮属于开发比较，不能替代冻结后的8个sealed主体确认。

{_table(primary)}

## 评分与统计口径

最终推荐由算法根据可见观测选择，报告只对这个推荐评分。完成且合格时selection_loss=true_E3；完成但不合格时为max(1,true_E3)；未完成、失败或无可评分推荐时为1。失败单列，不能用评估器在已执行集合中用真值重新挑选候选。较低损失更好，参考E3=1；improvement_pct=100×(1−selection_loss)，是应用不合格/失败惩罚后的改善，须与真实合格率一起解读。

先在每主体内对种子求均值，再在主体间等权汇总；Random的多种子不增加其主体权重。constraint_violations与invalid_trials为每次顺序运行累计次数的主体等权均值；constraint_violations是观测判定的超限次数，有噪时不等同于全部试验的真实超限次数，最终推荐的真实合格率则由独立评估器计算。K是含参考的预算上限，实际次数见subject_summary/summary中的executed_trials；Reference可能只执行一次。原始E3/E2/峰值的均值仅在有可评分推荐的记录上计算，不能替代包含失败的selection_loss。

elapsed_s为完整最长预算运行的wall time，包含模拟/缓存命中先后及模型拟合，同一运行的各预算行沿用该值；它仅供工程诊断，不能用于算法速度排名或较短预算耗时推断。

development方法差值以同主体均值配对，固定seed={BOOTSTRAP_SEED}、{BOOTSTRAP_REPEATS}次主体bootstrap，给出95%百分位区间。差值方向为A−B，selection_loss负值有利于A，真实合格率正值有利于A。只有一个主体时不计算主体区间。所有逐对结果在pairwise_comparisons.csv；以下列出MI-EI相关的无噪损失比较。未执行oracle全景分析，因此本报告不报告oracle regret。

{_table(paired_main)}

## Native 噪声与工程验证

{_table(native_table)}

Native为单一模型。噪声图的95%区间仅描述合成测量/算法重复，不是跨主体推断；无噪确定性方法只有一次时不绘制区间。种子数量以上表及protocol为准，本轮工程重复不能替代10月计划的正式30种子稳健性实验。1%/3%为参考关节RMS尺度的合成采样噪声，不代表实测传感器误差。

## 失败与限制

最终预算记录中的失败运行数：{len(failure_rows)}/{len(final_rows)}。下表按明确status保留失败；约束超限次数在主表及原始逐预算结果中保留，不因回退参考而抹除。

{_table(failures)}

两种候选族、相同预算、模型/端点/负荷约束下比较，不以未知真值筛选候选。有效但观测超限的样本保留用于拟合，最终推荐只从观测合格的已执行候选中选择；超限不补预算。E3及这些负荷约束只具有当前仿真机械语义；不支持临床疗效、舒适度、患者代表性、真实机器人安全或在线个体化结论。24个development已经用于开发，不能写成保留集；8个sealed主体仍须按原政策在最终冻结及另行授权后独立评估。没有明确赢家时如实保留并列或“无优势”，不通过改变模型制造算法优势。

## 文件与本次配置

- results.csv：每主体/方法/种子/预算的原始推荐成绩。
- subject_summary.csv：先在主体内汇总种子；summary.csv：再对主体等权汇总。
- pairwise_comparisons.csv：K4/K8的主体配对损失与合格率差值、95%区间。
- convergence.png：无噪最终推荐的预算曲线；noise_robustness.png：native噪声稳健性。
- protocol.json：完整运行设置与代码来源；以下原样摘录可用的关键配置。
- analysis_provenance.json：当前报告源码、运行前报告源码、protocol、completion、results及存在时trial_history的SHA；记录仅修正数值容差、完整性检查与主次预算标签，模拟、推荐及原始结果保持不变。

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

复现报告：`python -m lower_limb_sim.myoleg_benchmark.report --output-dir {output.as_posix()}`。此命令只重建汇总和图表。
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    (output / "analysis_provenance.json").write_text(
        json.dumps(analysis_provenance, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return dict(output_dir=str(output), result_rows=len(data), runs=len(final_rows),
                development_subjects=int(development.subject_id.nunique()),
                native_subjects=int(native.subject_id.nunique()), failure_runs=len(failure_rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize completed MyoLeg runs without importing a simulator")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate_report(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
