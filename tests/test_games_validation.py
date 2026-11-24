import pandas as pd
import pytest
from conftest import FIXTURES, make_games, needs_raw_subset

from nba_sos.config import PROJECT_ROOT
from nba_sos.ingestion.raw_store import RawStore
from nba_sos.ingestion.venues import build_venue_table, load_venue_config, parse_wkt_point
from nba_sos.preprocessing.games import build_games, build_schedule, excluded_game_counts, load_standings
from nba_sos.validation import checks as C

STORE = RawStore(FIXTURES / "raw_subset")


@pytest.fixture(scope="module")
def venue_cfg():
    return load_venue_config(PROJECT_ROOT / "configs")


@needs_raw_subset
def test_build_games_regular_season_only_and_neutral_detection(venue_cfg):
    aliases, venues = venue_cfg
    g = build_games(STORE, "2023-24", aliases, venues).set_index("game_id")
    assert set(g.index) == {"0022300172", "0022301229", "0022300001", "0022300002"}  # 006 and 001 excluded
    assert excluded_game_counts(STORE, "2023-24") == {"001": 1, "002": 4, "006": 1}
    # source does not flag these as neutral in 2023-24; venue table does
    assert not g.loc["0022300172", "source_is_neutral"] and g.loc["0022300172", "is_neutral"]
    assert g.loc["0022301229", "venue_id"] == "LAS_VEGAS_TMOBILE" and g.loc["0022301229", "is_neutral"]
    assert not g.loc["0022300001", "is_neutral"]


@needs_raw_subset
def test_possessions_are_mean_of_both_teams_and_ratings_consistent(venue_cfg):
    g = build_games(STORE, "2023-24", *venue_cfg)
    assert (g.possessions == (g.home_poss_source + g.away_poss_source) / 2).all()
    assert ((g.home_off_rating - g.away_off_rating - g.margin / g.possessions * 100).abs() < 1e-9).all()
    assert (g.home_plus_minus == g.margin).all()


@needs_raw_subset
def test_home_team_comes_from_schedule_when_matchup_says_away_for_both(venue_cfg):
    g = build_games(STORE, "2024-25", *venue_cfg).set_index("game_id")
    r = g.loc["0022400147"]  # Mexico City: LeagueGameLog writes 'MIA @ WAS' and 'WAS @ MIA'
    assert r.home_team == "WAS" and r.away_team == "MIA" and r.is_neutral
    assert not r.home_matchup_says_home
    assert (g._pair == "both").all()


@needs_raw_subset
def test_schedule_rows_have_divisions_coordinates_and_local_time(venue_cfg):
    aliases, venues = venue_cfg
    g = build_games(STORE, "2023-24", aliases, venues)
    s = build_schedule(g, load_standings(STORE, "2023-24"), venues)
    assert len(s) == 2 * len(g)
    assert s.conference.notna().all() and s.arena_latitude.notna().all()
    mex = s[s.game_id == "0022300172"].iloc[0]
    assert mex.venue_timezone == "America/Mexico_City"
    assert set(s.columns) >= {"team", "opponent", "home_away", "division", "tipoff_local_time"}


def test_venue_table_builder_rejects_ambiguous_and_missing():
    aliases = pd.DataFrame(
        {
            "arena_name": ["A", "B"],
            "arena_city": ["x", "y"],
            "venue_id": ["V", "V"],
            "wikidata_qid": ["Q1", "Q2"],
            "neutral_site": [False, False],
        }
    )
    with pytest.raises(ValueError, match="several Wikidata items"):
        build_venue_table(aliases, pd.DataFrame(), "t")
    aliases.loc[1, "wikidata_qid"] = "Q1"
    with pytest.raises(ValueError, match="no Wikidata coordinates"):
        build_venue_table(
            aliases,
            pd.DataFrame({"wikidata_qid": ["Q9"], "wikidata_label": ["z"], "latitude": [1.0], "longitude": [1.0]}),
            "t",
        )
    assert parse_wkt_point("Point(-118.343102 33.945127)") == (33.945127, -118.343102)


def test_every_alias_maps_to_a_venue_with_coordinates(venue_cfg):
    aliases, venues = venue_cfg
    assert set(aliases.venue_id) == set(venues.venue_id)
    assert venues[["latitude", "longitude", "timezone"]].notna().all().all()
    assert not aliases.duplicated(["arena_name", "arena_city"]).any()


# ------------------------------------------------------------------ check_games
def _by_name(results):
    return {r.check: r for r in results}


def full_round_robin():
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows, day = [], 1
    for h in teams:
        for a in teams:
            if h != a:
                rows.append(("2022-23", f"2022-11-{day:02d}", h, a, 100 + day, 90))
                day += 1
    return make_games(rows)


def test_check_games_clean_data_passes_except_team_count():
    res = _by_name(C.check_games(full_round_robin(), "2022-23"))
    assert not res["team_count"].passed  # synthetic league has 4 teams
    for name in [
        "game_id_unique",
        "two_team_rows_per_game",
        "equal_games_per_team",
        "game_count",
        "wl_matches_score",
        "plus_minus_matches_margin",
        "no_ties",
        "one_game_per_team_per_date",
        "home_matches_schedule",
        "possessions_present",
    ]:
        assert res[name].passed, name


@pytest.mark.parametrize(
    "mutate,check",
    [
        (lambda g: pd.concat([g, g.iloc[[0]]]), "game_id_unique"),
        (lambda g: g.assign(_pair=["left_only"] + ["both"] * (len(g) - 1)), "two_team_rows_per_game"),
        (lambda g: g.assign(home_wl=["L"] + list(g.home_wl[1:])), "wl_matches_score"),
        (lambda g: g.assign(home_score=pd.array([90] + list(g.home_score[1:]), dtype="Int64")), "no_ties"),
        (lambda g: g.assign(home_plus_minus=[0] + list(g.home_plus_minus[1:])), "plus_minus_matches_margin"),
        (lambda g: g.assign(game_date=[g.game_date.iloc[1]] + list(g.game_date[1:])), "one_game_per_team_per_date"),
        (lambda g: g.iloc[1:], "equal_games_per_team"),
        (
            lambda g: g.assign(sched_home_team_id=pd.array([3] + list(g.sched_home_team_id[1:]), dtype="Int64")),
            "home_matches_schedule",
        ),
    ],
)
def test_check_games_detects_errors(mutate, check):
    g = mutate(full_round_robin())
    assert not _by_name(C.check_games(g, "2022-23"))[check].passed
