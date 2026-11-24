"""Generate reports/data_audit.md from the build outputs."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df is None or len(df) == 0:
        return "_（无）_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def out_consistency(availability: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Share of pre-game 'Out' players (matched to an id) who indeed logged no minutes.
    Audit only; uses post-game box scores, so never a feature."""
    a = availability[availability.player_id.notna()]
    played = player_games[player_games.minutes > 0][["game_id", "player_id"]].assign(played=True)
    m = a.merge(played, on=["game_id", "player_id"], how="left")
    m["played"] = m.played.astype("boolean").fillna(False).astype(bool)
    return m.groupby("injury_status").agg(n=("played", "size"), share_played=("played", "mean")).reset_index()


def write_audit(
    paths,
    cfg,
    *,
    store,
    games,
    all_games,
    schedule,
    checks,
    excluded,
    venues,
    aliases,
    counts,
    deviations,
    selection,
    parse_info,
    parsed,
    availability,
    team_status,
    player_games,
    features,
    sos,
    backtest,
    predictions,
) -> None:
    L: list[str] = []
    w = L.append
    w("# 数据审计报告（Phase 1）\n")
    w(f"生成时间（UTC）：{datetime.now(UTC).isoformat(timespec='seconds')}\n")
    w("本报告由 `python -m nba_sos.pipeline build` 自动生成，所有数字来自 `data/raw/` 中的原始快照。\n")

    # 1 sources
    w("## 1. 数据来源与抓取时间\n")
    man = pd.DataFrame(store.read_manifest())
    stats = man[man.source == "stats.nba.com"]
    if len(stats):
        w(
            _md(
                stats[["name", "season", "fetched_at", "bytes"]]
                .drop_duplicates(["name", "season"], keep="last")
                .sort_values(["name", "season"])
            )
        )
    w(
        "\n上游接口不提供逐条记录的更新时间，`fetched_at` 是我们抓取的时间，无法据此还原数据修订历史。"
        "历史赛程是最终实际赛程（含改期后日期、NBA Cup 淘汰赛对阵），不是赛季开始时公布的版本。\n"
    )

    # 2 coverage
    w("## 2. 覆盖范围\n")
    cov = (
        all_games.groupby("season")
        .agg(
            games=("game_id", "size"),
            first=("game_date", "min"),
            last=("game_date", "max"),
            neutral=("is_neutral", "sum"),
            source_neutral=("source_is_neutral", "sum"),
        )
        .reset_index()
    )
    cov["teams"] = [
        pd.concat(
            [all_games[all_games.season == s].home_team_id, all_games[all_games.season == s].away_team_id]
        ).nunique()
        for s in cov.season
    ]
    w(_md(cov))
    w("\n2020-21 只用于计算 2021-22 的赛季前评级（72 场缩水赛季，多数比赛无观众，不用于估计主场优势）。\n")
    w(
        "\n赛程中被排除的非常规赛比赛（按 game_id 前缀：001 季前赛、002 常规赛、003 全明星、004 季后赛、"
        "005 附加赛、006 NBA Cup 决赛）：\n"
    )
    ex = pd.DataFrame([{"season": s, **{f"prefix_{k}": v for k, v in d.items()}} for s, d in excluded.items()])
    w(_md(ex.fillna(0).astype({c: int for c in ex.columns if c != "season"})))

    # 3 validation
    w("## 3. 验证结果\n")
    bad = checks[~checks.passed]
    w(
        f"共 {len(checks)} 项检查，未通过 {len(bad)} 项"
        f"（错误 {int((bad.severity == 'error').sum())}，警告 {int((bad.severity == 'warning').sum())}）。"
        "完整结果见 `reports/validation_results.csv`。\n"
    )
    w("\n未通过或需关注的项目：\n")
    attention = checks[
        ~checks.passed | checks.check.isin(["possession_source_agreement", "neutral_flag_source_vs_venue"])
    ]
    w(_md(attention[["check", "season", "severity", "passed", "detail"]]))

    # 4 neutral & venues
    w("## 4. 中立场地与场馆\n")
    nn = games[games.is_neutral][
        [
            "season",
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "arena_name",
            "arena_city",
            "source_is_neutral",
            "game_label",
            "game_sublabel",
        ]
    ]
    w(
        "`is_neutral` = 源数据 `isNeutral` 或场馆被标为中立场地（国际赛、拉斯维加斯 NBA Cup）。"
        "源数据在 2024-25 之前没有标记这些比赛。\n"
    )
    w(_md(nn))
    w(
        f"\n场馆：{aliases.shape[0]} 个名称/城市组合映射到 {venues.shape[0]} 座实体场馆"
        "（`configs/venue_aliases.csv` 人工整理，坐标来自 Wikidata P625，时区由坐标离线推算）。\n"
    )
    w(_md(venues[["venue_id", "wikidata_qid", "latitude", "longitude", "timezone", "neutral_site"]]))

    # 5 matchups
    w("## 5. 对阵次数矩阵\n")
    rs = counts.groupby(["season", "team"]).n_games.sum()
    w(
        f"每队对阵合计：最小 {rs.min()}，最大 {rs.max()}。与传统 4/4/3/2 结构不一致的球队-关系组合："
        f"{len(deviations)} 个。\n"
    )
    if len(deviations):
        w(
            _md(
                deviations.groupby(["season", "relation", "actual", "traditional"])
                .size()
                .rename("n_teams")
                .reset_index()
            )
        )

    # 6 injuries
    w("## 6. 伤病报告\n")
    sel = selection.merge(games[["game_id", "season"]], on="game_id")
    cov_i = (
        sel.groupby("season")
        .agg(games=("game_id", "size"), with_report=("report_path", lambda s: s.notna().sum()))
        .reset_index()
    )
    ts = (
        team_status.groupby("season")
        .agg(
            team_games=("game_id", "size"),
            usable=("snapshot_usable", "sum"),
            not_submitted=("not_submitted", "sum"),
            game_not_in_report=("game_in_report", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    cov_i = cov_i.merge(ts, on="season", how="outer")
    cov_i["usable_share"] = cov_i.usable / cov_i.team_games
    w(_md(cov_i))
    if len(sel.dropna(subset=["report_path"])):
        lead = games.set_index("game_id").tipoff_utc - sel.set_index("game_id").report_published_at_utc
        lead_min = lead.dropna().dt.total_seconds() / 60
        w(
            f"\n所选报告发布时间到开赛的间隔（分钟）：中位数 {lead_min.median():.0f}，"
            f"最小 {lead_min.min():.0f}，最大 {lead_min.max():.0f}。所有所选报告均早于开赛。\n"
        )
    if parse_info is not None and len(parse_info):
        perr = parse_info.parse_error.notna().sum()
        w(
            f"\n解析报告文件 {len(parse_info)} 份，解析失败 {perr} 份；"
            f"无法归位的换行碎片合计 {int(parse_info.orphan_fragments.fillna(0).sum())} 个。\n"
        )
        if perr:
            w(_md(parse_info[parse_info.parse_error.notna()][["report_path", "parse_error"]].head(20)))
    if len(availability):
        mr = (
            availability.match_method.fillna("unmatched")
            .value_counts()
            .rename_axis("match_method")
            .rename("n")
            .reset_index()
        )
        mr["share"] = mr.n / mr.n.sum()
        w("\n球员名匹配到 nba_api 球员 ID：\n")
        w(_md(mr))
        w(
            "\n未匹配的球员大多是整季未出场的球员（双向合同、长期伤病），他们此前的上场时间本来就是 0。"
            "匹配时用到了该赛季完整的球员名单，这只是身份识别，不进入任何特征值。\n"
        )
        inv = availability[~availability.status_valid.astype(bool)]
        w(f"\n状态不在 {{Out, Doubtful, Questionable, Probable, Available}} 内的行：{len(inv)}。\n")
        oc = out_consistency(availability, player_games)
        w("\n一致性审计（赛后核对，不作为特征）：赛前各状态球员最终实际上场的比例：\n")
        w(_md(oc))

    # 7 features
    w("## 7. 赛前特征\n")
    num = features.select_dtypes("number").columns.tolist() + ["is_back_to_back"]
    nul = features[num].isna().mean().rename("null_share").reset_index().rename(columns={"index": "feature"})
    w(_md(nul))
    w(
        "\n结果类特征只用比赛日期（ET）严格早于当天的比赛；每季第一场的累计值和休息天数为空，不填补。"
        "伤病类特征在没有可用赛前快照时为空，不按 0 填补。"
        "`as_of_timestamp` = max(此前日期最晚一场比赛开赛 + 4 小时, 所用伤病报告发布时间)，构建时检查它早于开赛。\n"
    )
    desc = (
        features[
            [
                "rest_days",
                "games_last_4_days",
                "games_last_7_days",
                "travel_distance_km",
                "timezone_change_hours",
                "consecutive_road_games",
                "missing_minutes_prior_avg",
            ]
        ]
        .describe()
        .T
    )
    w(_md(desc.reset_index().rename(columns={"index": "feature"}), ".2f"))

    # 8 sos
    w("## 8. 静态 SOS\n")
    w(
        "`retro_*` 用对手全季最终战绩，仅作描述；`causal_*` 用对手截至赛前的战绩，是可用于预测的版本。"
        "排名 1 = 赛程最难。每季前 5 名（按 retro_owp）：\n"
    )
    top = sos.sort_values(["season", "retro_owp_rank"]).groupby("season").head(5)
    w(
        _md(
            top[
                [
                    "season",
                    "team",
                    "win_pct",
                    "retro_owp",
                    "retro_owp_rank",
                    "causal_owp_mean",
                    "causal_owp_mean_rank",
                    "retro_opp_net_rating",
                    "causal_opp_net_rating_mean",
                ]
            ]
        )
    )
    corr = sos.groupby("season").apply(
        lambda d: d.retro_owp.corr(d.causal_owp_mean, method="spearman"), include_groups=False
    )
    w("\nretro 与 causal OWP 的 Spearman 相关：" + "，".join(f"{s} {v:.2f}" for s, v in corr.items()) + "\n")

    # 9 baseline
    w("## 9. 基线模型：赛季前评级 + 主场优势\n")
    w(
        "扩展窗口滚动回测：每个测试赛季只用更早的赛季训练。`b` 是上季净效率差到本季净胜分的系数，"
        "`hca` 是主场优势（分），`sigma` 是训练残差标准差。\n"
    )
    cols = [
        "test_season",
        "train_seasons",
        "b",
        "hca",
        "sigma",
        "n_games",
        "preseason_log_loss",
        "constant_log_loss",
        "preseason_brier",
        "constant_brier",
        "preseason_margin_mae",
        "constant_margin_mae",
        "preseason_margin_rmse",
        "constant_margin_rmse",
        "preseason_ece",
    ]
    w(_md(backtest[[c for c in cols if c in backtest.columns]]))
    from ..evaluation.metrics import calibration_table

    cal = calibration_table(predictions.home_win.to_numpy(float), predictions.pred_p_home.to_numpy(float))
    w("\n合并测试集的校准表：\n")
    w(_md(cal))

    (paths.reports / "data_audit.md").write_text("\n".join(L), encoding="utf-8")
