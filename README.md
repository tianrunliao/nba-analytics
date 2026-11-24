# NBA 动态赛程难度与数据分析（D-SOS）

从 2007-08 起 19 个赛季的官方数据出发，按"只用开赛前可获得的信息"的原则，构建球队动态实力、单场预测、伤病与阵容、剩余赛季模拟和赛程难度（D-SOS），并在此基础上做季后赛纯实力、球员价值等分析。

| 阶段 | 内容 | 主要代码 |
|---|---|---|
| Phase 1 | 数据管道、数据验证、赛前特征（回合、效率、赛程密度、旅行、背靠背情境、伤病报告）、静态 SOS、季前基线 | `ingestion/`、`preprocessing/`、`validation/`、`features/`、`sos/static.py` |
| Phase 2 | Elo、EWMA、动态对手调整实力、CatBoost 单场模型，滚动回测与消融 | `ratings/elo.py`、`ratings/dynamic.py`、`models/game_model.py` |
| Phase 3 | 出场概率、阵容缺失价值、重复计量检验、复出时间表 | `models/availability.py` |
| Phase 4 | 平均球队反事实 SOS 与 Shapley 拆解、全联盟蒙特卡洛（实力不确定性、阵容先验、未来伤病、官方平手规则、NBA 杯）、历史日期回放 | `sos/projection.py`、`sos/availability.py`、`sos/roster_prior.py`、`sos/standings.py`、`sos/cup.py`、`models/season_simulation.py` |
| 分析 | 球员价值（技术统计先验 + 五人阵容 RAPM）、季后赛纯实力与改进、季后赛升降球员、球星与深度 | `ratings/` |

所有模型选择只在测试赛季之前的赛季里进行（按赛季分组的 k-fold），测试赛季只做按时间顺序的回放。`reports/` 是这些步骤的输出结果。

## 运行

```bash
uv sync
.venv/bin/python -m nba_sos.pipeline fetch            # 联网：nba_api 原始数据 + 当前赛季赛程快照
.venv/bin/python -m nba_sos.pipeline fetch-injuries   # 联网：官方伤病报告 PDF（约 2–3 小时）
.venv/bin/python -m nba_sos.pipeline build            # 离线：原始数据 → 清洗表、特征、SOS、基线、审计报告
.venv/bin/python -m nba_sos.pipeline snapshot         # 联网：当前赛季赛程与已完成比赛的带时间戳快照
.venv/bin/python -m nba_sos.pipeline window-sos --n 15 # 当前赛季每队前 n 场的赛程难度（需先 build + snapshot）
.venv/bin/python -m nba_sos.pipeline playoff-rank      # 季后赛纯实力榜单 + 滚动回测（需先 build）
.venv/bin/python -m nba_sos.pipeline analysis          # 球员价值、模型变体对比、球星 vs 深度、常规赛前五球员
.venv/bin/python -m nba_sos.pipeline game-model        # Phase 2：Elo / EWMA / 动态实力 / CatBoost 滚动回测与消融
.venv/bin/python -m nba_sos.pipeline availability      # Phase 3：出场概率、阵容贡献、重复计量检验、伤病模块评估、复出时间表
.venv/bin/python -m nba_sos.pipeline playoff-improve   # 用历届季后赛嵌套选择改进季后赛模型（含季后赛内状态更新）
.venv/bin/python -m nba_sos.pipeline replay            # Phase 4：历史日期回放，检验剩余赛季模拟（胜场误差、区间覆盖率）
.venv/bin/python -m nba_sos.pipeline project           # Phase 4：本赛季剩余赛程 D-SOS、预期胜场、种子与附加赛概率（先运行 snapshot）
.venv/bin/python -m nba_sos.pipeline playoff-risers    # 季后赛相对常规赛变强/变弱的球员（同口径官方数据，含对手与样本量收缩）
.venv/bin/python -m pytest
```

`build` 只读 `data/raw/`，可以重复执行；验证出现错误级问题时，会先写出审计报告，然后以非零状态退出。

## 数据来源

| 来源 | 内容 | 位置 |
|---|---|---|
| stats.nba.com（nba_api 1.12.0） | 比赛日志、球队 Advanced（POSS 等）、球员比赛日志、赛程、排名 | `data/raw/nba_stats/` |
| NBA 官方伤病报告 PDF | 赛前快照，HTTP `Last-Modified` 作为发布时间 | `data/raw/injury_reports/` |
| Wikidata P625 | 场馆坐标（人工复核 QID） | `configs/venue_aliases.csv`、`configs/venues.csv` |

每个原始文件都在 `data/raw/manifest.jsonl` 里记录了 URL、抓取时间和 sha256。数据来自 stats.nba.com 和 NBA 官网，使用时须遵守 NBA 的使用条款，仅用于个人研究。

本仓库不包含原始数据和清洗后的数据（`data/` 约 660 MB，且 NBA 条款限制再分发），需要用上面的 `fetch` 和 `build` 命令在本地重建；解析真实 NBA 文件的测试夹具同样不上传，缺失时对应测试会自动跳过，重建方法见 `tests/fixtures/README.md`。`reports/raptor_comparison.csv` 是与 FiveThirtyEight RAPTOR（CC BY 4.0，https://github.com/fivethirtyeight/data/tree/master/nba-raptor）的相关性对比，只保留了相关系数。

## 输出

| 文件 | 内容 |
|---|---|
| `data/processed/games.parquet` | 每场一行：比分、回合数、攻防效率、中立场地、场馆 |
| `data/processed/schedule.parquet` | 每队每场一行：对手、主客、联盟/分区、场馆坐标、当地开赛时间 |
| `data/processed/team_game_features.parquet` | 每队每场的赛前特征（效率、赛程密度、旅行、伤病）及 `as_of_timestamp` |
| `data/processed/player_availability.parquet` | 每场开赛前最后一份伤病报告中的球员状态 |
| `data/processed/matchup_counts.parquet` | 实际对阵次数 N_ij |
| `reports/data_audit.md` | 数据审计报告 |
| `reports/static_sos.csv`、`reports/baseline_backtest.csv`、`reports/validation_results.csv` | 静态 SOS、基线回测、全部验证结果 |

## Phase 1 的关键口径

| 项目 | 口径 |
|---|---|
| 赛季范围 | 只取常规赛（ID 前缀 002），含 NBA Cup 小组赛和计入战绩的淘汰赛；Cup 决赛（006）、附加赛、季后赛排除 |
| 回合数 | 两队 POSS 的平均值；效率 = 得分 / 回合数 × 100 |
| 主客场 | 以赛程的主队 ID 为准（中立场地比赛的 MATCHUP 两队都写 "@"） |
| 中立场地 | 源数据 `isNeutral` 或场馆标为中立（国际赛、拉斯维加斯）；基线模型中主场优势记 0 |
| 信息截止 | 结果类特征只用 ET 日期严格早于当天的比赛；伤病只用发布时间早于开赛的最后一份报告 |
| 赛季首场旅行 | 从本队主场场馆出发 |
| 背靠背情境 | `b2b_type`（前一场+本场的主客，如 AA = 连续两场客场；中立场地算客场）、`turnaround_hours`（两场开赛间隔）、`prev_overtime_periods`、两场的场馆当地开赛时间、本队主场时区下的开赛时间（生物钟），与 `travel_distance_km`、`timezone_change_hours`（正 = 向东）组合使用 |
| 缺失 | 每季首场的累计值、无可用伤病快照的伤病特征保持为空，不填补 |
| 赛季前评级 | 上季每百回合净效率；与本季净胜分的系数在训练赛季上用 OLS 估计 |
| 静态 SOS | `retro_*` 用对手全季最终战绩（仅描述）；`causal_*` 用对手赛前战绩 |
