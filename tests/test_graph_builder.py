"""Gate-2 graph-builder tests (docs/SCHEMA.md invariants + the rugby fixture).

The rugby table is Figure 1 of the GRAMS+ paper. We back it with a deterministic
in-memory FakeKG (no network) encoding the entities' types, hierarchy and
statements, and the HashingEncoder for structure-only (non-learned) features.
"""
from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from flint.data.kg import FakeKG  # noqa: E402
from flint.graph.builder import (  # noqa: E402
    TableInput,
    build_hetero_graph,
    extract_ontology_neighborhood,
)
from flint.graph.features import HashingEncoder  # noqa: E402

FIX = pathlib.Path(__file__).parent / "fixtures" / "rugby_table.json"


def load_fixture():
    return json.loads(FIX.read_text())


# Players (col 0) are humans; teams (col 2) are national rugby union teams.
# Each player is a member (P54) of their team.
PLAYER_Q = {
    "Dan Carter": "Q_dc",
    "Ronan O'Gara": "Q_ro",
    "Percy Montgomery": "Q_pm",
    "Chris Paterson": "Q_cp",
}
TEAM_Q = {
    "New Zealand": "Q_nz",
    "Ireland": "Q_ie",
    "South Africa": "Q_za",
    "Scotland": "Q_sco",
}
# member-of-sports-team statements, paired in fixture row order
_MEMBER = dict(zip(PLAYER_Q.values(), TEAM_Q.values()))
RUGBY_KG = FakeKG(
    instance_of={
        **{q: ["Q5"] for q in PLAYER_Q.values()},
        **{q: ["Q58840819"] for q in TEAM_Q.values()},
    },
    subclass_of={
        "Q5": ["Q215627"],          # human -> person  (the expected ancestor chain)
        "Q215627": ["Q_agent"],
        "Q58840819": ["Q12973014"],  # national rugby union team -> sports team
    },
    statements={p: [("P54", t)] for p, t in _MEMBER.items()},
    labels={"Q5": "human", "Q215627": "person", "Q58840819": "national rugby union team"},
)


def _rugby_table(rows: list[list[str]]) -> TableInput:
    fx = load_fixture()
    # seed candidates from the player/team name in each entity column (1 per cell)
    candidates = []
    for col in fx["entity_columns"]:
        col_cands = []
        for row in rows:
            name = row[col]
            qid = PLAYER_Q[name] if col == 0 else TEAM_Q[name]
            col_cands.append([qid])
        candidates.append(col_cands)
    return TableInput(
        table_id=fx["table_id"],
        headers=fx["headers"],
        rows=rows,
        candidates=candidates,
        entity_columns=fx["entity_columns"],
    )


def _build(rows):
    table = _rugby_table(rows)
    cand_types = {"Q5", "Q58840819"}
    onto = extract_ontology_neighborhood(cand_types, max_hops=3, kg_index=RUGBY_KG)
    pv: dict[str, int] = {}
    return build_hetero_graph(
        table, onto, feature_encoder=HashingEncoder(32), kg=RUGBY_KG, prop_vocab=pv
    ), pv


def test_fixture_is_well_formed():
    fx = load_fixture()
    assert len(fx["headers"]) == 4
    assert all(len(r) == 4 for r in fx["rows"])
    assert fx["entity_columns"] == [0, 2]


def test_rugby_graph_has_expected_candidate_edges():
    fx = load_fixture()
    g, pv = _build(fx["rows"])

    # 8 candidates (4 players + 4 teams), one per entity cell
    assert g["cell_candidate"].num_nodes == 8
    # SCHEMA invariant 2: exactly one in_column edge per candidate
    inc = g["cell_candidate", "in_column", "column"].edge_index
    assert inc.shape[1] == 8
    assert sorted(inc[0].tolist()) == list(range(8))

    # kg_triple P54 edges: each player -> their team (4 of them)
    kt = g["cell_candidate", "kg_triple", "cell_candidate"]
    assert "P54" in pv
    p54 = pv["P54"]
    n_p54 = int((kt.edge_attr == p54).sum())
    assert n_p54 == 4
    # every kg_triple edge runs from a player (col 0) to a team (col 2)
    col = g["cell_candidate"].col
    for e in range(kt.edge_index.shape[1]):
        s, d = kt.edge_index[0, e].item(), kt.edge_index[1, e].item()
        assert col[s].item() == 0 and col[d].item() == 2

    # instance_of edges connect candidates to onto_class
    iof = g["cell_candidate", "instance_of", "onto_class"].edge_index
    assert iof.shape[1] == 8  # each candidate has exactly one type here

    # Q5 -> Q215627 ancestor chain present via subclass_of, with hop attr
    classes = g["onto_class"].class_ids
    assert "Q5" in classes and "Q215627" in classes
    sc = g["onto_class", "subclass_of", "onto_class"]
    ci = {c: i for i, c in enumerate(classes)}
    found = False
    for e in range(sc.edge_index.shape[1]):
        s, d = sc.edge_index[0, e].item(), sc.edge_index[1, e].item()
        if s == ci["Q5"] and d == ci["Q215627"]:
            assert sc.edge_attr[e].item() == 1  # person is 1 hop above human
            found = True
    assert found, "Q5 -> Q215627 subclass_of edge missing"


def test_subclass_edges_form_a_dag():
    """SCHEMA invariant 4: no cycles in the attached subclass-of neighborhood."""
    fx = load_fixture()
    g, _ = _build(fx["rows"])
    sc = g["onto_class", "subclass_of", "onto_class"].edge_index
    n = g["onto_class"].num_nodes
    adj = {i: [] for i in range(n)}
    for e in range(sc.shape[1]):
        adj[sc[0, e].item()].append(sc[1, e].item())
    color = [0] * n  # 0=white,1=gray,2=black

    def has_cycle(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1 or (color[v] == 0 and has_cycle(v)):
                return True
        color[u] = 2
        return False

    assert not any(has_cycle(u) for u in range(n) if color[u] == 0)


def _edge_signature(g, pv):
    """Content-keyed edge multisets, invariant under node renumbering."""
    ent = g["cell_candidate"].entity_ids
    col = g["cell_candidate"].col.tolist()
    cls = g["onto_class"].class_ids
    inv_pv = {v: k for k, v in pv.items()}
    sig = {}
    inc = g["cell_candidate", "in_column", "column"].edge_index
    sig["in_column"] = sorted((ent[inc[0, e]], inc[1, e].item()) for e in range(inc.shape[1]))
    sr = g["cell_candidate", "same_row", "cell_candidate"].edge_index
    sig["same_row"] = sorted((ent[sr[0, e]], ent[sr[1, e]]) for e in range(sr.shape[1]))
    iof = g["cell_candidate", "instance_of", "onto_class"].edge_index
    sig["instance_of"] = sorted((ent[iof[0, e]], cls[iof[1, e]]) for e in range(iof.shape[1]))
    kt = g["cell_candidate", "kg_triple", "cell_candidate"]
    sig["kg_triple"] = sorted(
        (ent[kt.edge_index[0, e]], ent[kt.edge_index[1, e]], inv_pv[kt.edge_attr[e].item()])
        for e in range(kt.edge_index.shape[1])
    )
    sc = g["onto_class", "subclass_of", "onto_class"]
    sig["subclass_of"] = sorted(
        (cls[sc.edge_index[0, e]], cls[sc.edge_index[1, e]], sc.edge_attr[e].item())
        for e in range(sc.edge_index.shape[1])
    )
    return sig


def test_row_permutation_invariance():
    """SCHEMA invariant 1: shuffling rows yields an isomorphic graph."""
    fx = load_fixture()
    rows = fx["rows"]
    shuffled = [rows[i] for i in (2, 0, 3, 1)]
    g1, pv1 = _build(rows)
    g2, pv2 = _build(shuffled)
    # same node counts per type
    for nt in ["column", "cell_candidate", "onto_class", "onto_property"]:
        assert g1[nt].num_nodes == g2[nt].num_nodes
    # identical content-keyed edge multisets
    assert _edge_signature(g1, pv1) == _edge_signature(g2, pv2)
