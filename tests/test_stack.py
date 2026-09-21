"""The `stack` branch: question DAGs in row form, the UNKNOWN dustbin key, and the evidence pointer.

Run: uv run --extra serve python -m pytest tests/test_stack.py -q
"""
import json
import sys
from pathlib import Path

import pytest
import torch

from kev.api import SystemOneRequest, split_unknown, to_answers, to_record
from kev.benchmark import validate_distribution
from kev.composition import POLICY_WRAPPERS, evaluate_rule, render_rule, rendered_facts
from kev.dag import expand, strip_deps
from kev.data import materialize
from kev.model import PointerHead, dependency_closure, encode, load_tokenizer, rows_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_stack_data as builder                      # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer("Qwen/Qwen2.5-0.5B")


def plain_record():
    return {"state": "Order 4411 arrived late and the box was crushed. Two charges appear on the card.",
            "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0},
                          {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing"], "label": 2},
                          {"instr": "How upset is the customer?", "options": ["calm", "annoyed", "furious"], "label": 1}]}


def test_rows_without_deps_are_the_packed_slices(tok):
    """A record with no dependencies must keep exactly today's row form: the same ids, the same branch positions, and
    the same readout offsets. This is the bit-identity guarantee for every existing checkpoint."""
    enc = encode(tok, plain_record())
    state_ids, state_pos, rows = rows_of(enc)
    assert enc["deps"] == [(), (), ()] and len(rows) == 3
    Ls = enc["seg"].count(0)
    assert state_ids == enc["ids"][:Ls] and state_pos == enc["pos"][:Ls] == list(range(Ls))
    start = Ls
    for row, decide, opts in zip(rows, enc["decide_idx"], enc["opt_idx"]):
        end = decide + 1
        assert row["ids"] == enc["ids"][start:end]
        assert row["pos"] == enc["pos"][start:end] == list(range(Ls, Ls + len(row["ids"])))
        assert row["decide"] == decide - start and row["opts"] == [o - start for o in opts]
        start = end


def test_dependency_rows_carry_the_closure_in_topological_order(tok):
    """q3 depends on q2, q2 on q1: q3's row is state + q1 + q2 + q3, positions running from len(state)."""
    rec = plain_record()
    rec["questions"][1]["deps"] = [0]
    rec["questions"][2]["deps"] = [1]
    enc = encode(tok, rec)
    assert dependency_closure(enc["deps"], 2) == [0, 1]
    Ls = enc["seg"].count(0)
    _, _, rows = rows_of(enc)
    spans, start = [], Ls
    for d in enc["decide_idx"]:
        spans.append((start, d + 1)); start = d + 1
    assert rows[0]["ids"] == enc["ids"][spans[0][0]:spans[0][1]]               # no deps: unchanged
    for k, closure in ((1, [0]), (2, [0, 1])):
        expected = [t for j in closure for t in enc["ids"][spans[j][0]:spans[j][1]]] + list(enc["ids"][spans[k][0]:spans[k][1]])
        assert rows[k]["ids"] == expected
        assert rows[k]["pos"] == list(range(Ls, Ls + len(expected)))
        assert rows[k]["ids"][rows[k]["decide"]] == enc["ids"][enc["decide_idx"][k]]
        assert [rows[k]["ids"][o] for o in rows[k]["opts"]] == [enc["ids"][o] for o in enc["opt_idx"][k]]

    forward = plain_record(); forward["questions"][0]["deps"] = [1]
    with pytest.raises(ValueError, match="earlier questions"):
        encode(tok, forward)
    cycle = plain_record(); cycle["questions"][1]["deps"] = [1]
    with pytest.raises(ValueError, match="earlier questions"):
        encode(tok, cycle)
    # the same rule at the request level: `depends_on` names ids, and only earlier ones resolve
    body = {"state": "s", "questions": {"a": {"type": "noul", "instructions": "i", "depends_on": ["b"]},
                                        "b": {"type": "noul", "instructions": "i"}}}
    with pytest.raises(ValueError, match="earlier question"):
        to_record(SystemOneRequest.model_validate(body))


def test_dustbin_is_off_by_default_and_appends_one_key():
    torch.manual_seed(0)
    plain = PointerHead(8, dp=4)
    torch.manual_seed(0)
    withbin = PointerHead(8, dp=4, dustbin=True)
    h_decide, h_opts = torch.randn(8), torch.randn(3, 8)
    a, b = plain(h_decide, h_opts), withbin(h_decide, h_opts)
    assert a.shape == (3,) and b.shape == (4,)
    assert torch.equal(a, b[:3])                                        # the real options are untouched
    q = withbin.q(h_decide)
    assert torch.allclose(b[3], (withbin.null_key @ q) * withbin.scale + withbin.null_bias)
    assert float(withbin.null_bias.detach()) == -5.0 and float(b[3].detach()) == -5.0    # zero null key: the bias is the whole logit at init


def test_sentence_indices_map_to_tokens_and_drop_past_the_state_limit(tok):
    """Offsets are computed on the text the tokenizer actually sees, so a forged delimiter (rewritten to <¦x¦>) does not
    shift them; a sentence pushed past max_state is dropped and any question needing it loses its evidence target."""
    first = "The account <|box_start|> holder is Mira."
    last = "The request value is 12."
    state = first + " " + " ".join(["Filler sentence number %d." % i for i in range(400)]) + " " + last
    rec = {"state": state, "sentences": [{"text": first, "facts": ["holder"]}, {"text": last, "facts": ["request value"]}],
           "questions": [{"instr": "Who?", "options": ["a", "b"], "label": 0, "evidence": [0]},
                         {"instr": "How much?", "options": ["a", "b"], "label": 0, "evidence": [1]},
                         {"instr": "Both?", "options": ["a", "b"], "label": 0, "evidence": [0, 1]}]}
    enc = encode(tok, rec, max_state=384)
    assert enc["sentences_dropped"] == 1 and len(enc["sent_idx"]) == 1
    assert enc["evidence"] == [[0], None, None]
    decoded = tok.decode(enc["ids"][1:enc["sent_idx"][0] + 1])
    assert decoded.endswith("holder is Mira.") and "<¦box_start¦>" in decoded
    assert enc["sent_idx"][0] < enc["seg"].count(0)


def certificate_record():
    """A hand-built compositional record: (request value >= 10 OR account verified) AND case value < 5."""
    tree = ("and", ("or", 0, 1), 2)
    atoms = [{"kind": "ge", "fields": ["request value"], "threshold": 10},
             {"kind": "flag", "fields": ["account verified"], "threshold": 0},
             {"kind": "lt", "fields": ["case value"], "threshold": 5}]
    facts = {"routing reference": 700, "request value": 12, "account verified": False, "case value": 3}
    order = ["routing reference", "request value", "account verified", "case value"]
    label = evaluate_rule(tree, atoms, facts)
    assert label is True
    state = {"policy": POLICY_WRAPPERS[0].format(rule=render_rule(tree, atoms, 0)), "case": " ".join(rendered_facts(facts, order))}
    return {"state": state,
            "questions": {"decision": {"type": "choice", "instructions": "Apply the policy to this case.",
                                       "criteria": {"accept": "The policy permits this case", "reject": "The policy does not permit this case"},
                                       "label": "accept", "src": "composition_test"}},
            "_meta": {"source": "compositional", "variant": "clean", "id": "t/0", "group_id": "t/0", "render_style": 0,
                      "certificate": {"tree": tree, "atoms": atoms, "facts": facts, "order": order,
                                      "deciding_field": "case value", "label": label}}}


def test_evidence_targets_follow_the_rule_tree():
    out = expand(certificate_record())
    assert list(out["questions"]) == ["atom_0", "atom_1", "atom_2", "node_0", "decision"]
    sentences = [s["facts"] for s in out["_meta"]["sentences"]]
    assert sentences == [["routing reference"], ["request value"], ["account verified"], ["case value"]]
    evidence = {qid: q["evidence"] for qid, q in out["questions"].items()}
    assert evidence == {"atom_0": [1], "atom_1": [2], "atom_2": [3], "node_0": [1, 2], "decision": [1, 2, 3]}
    assert out["questions"]["node_0"]["deps"] == ["atom_0", "atom_1"]
    assert out["questions"]["decision"]["deps"] == ["node_0", "atom_2"]
    assert [q["label"] for q in out["questions"].values()] == [True, False, True, True, "accept"]
    # the routing reference carries no atom, so no question ever points at it
    assert all(0 not in q["evidence"] for q in out["questions"].values())
    assert all("deps" not in q for q in strip_deps(out)["questions"].values())
    rec = materialize(out)
    assert [q.get("deps") for q in rec["questions"]] == [None, None, None, [0, 1], [3, 2]]


@pytest.mark.parametrize("flavour", ["dustbin", "uniform"])
def test_dropping_the_deciding_sentence_marks_only_what_it_decides(flavour):
    record = certificate_record()
    cert = record["_meta"]["certificate"]
    intact = expand(record)
    facts = {k: v for k, v in cert["facts"].items() if k != cert["deciding_field"]}
    out = expand(record, facts=facts)
    assert cert["deciding_field"] not in out["state"]["case"] and "case value" not in out["state"]["case"]
    labels = {qid: q["label"] for qid, q in out["questions"].items()}
    assert labels["atom_0"] is True and labels["atom_1"] is False and labels["node_0"] is True   # still determined
    assert labels["atom_2"] is None and labels["decision"] is None                               # the missing fact decides these
    marked = {qid: (builder.mark_unknown(q, intact["questions"][qid]["label"], flavour) if q["label"] is None else q)
              for qid, q in out["questions"].items()}
    rec = materialize({**out, "questions": marked})
    by_id = {q["qid"]: q for q in rec["questions"]}
    if flavour == "dustbin":
        assert by_id["atom_2"]["unknown"] and by_id["atom_2"]["label"] == len(by_id["atom_2"]["options"])
        assert by_id["decision"]["label"] == len(by_id["decision"]["options"])
        assert "unknown" not in by_id["atom_0"] and by_id["atom_0"]["label"] == 1
    else:
        assert by_id["atom_2"]["target"] == [0.5, 0.5] and by_id["decision"]["target"] == [0.5, 0.5]
        assert by_id["atom_0"].get("target") is None


def test_to_answers_renormalises_and_reports_the_dustbin():
    _, meta = to_record(SystemOneRequest.model_validate({"state": "s", "questions": {
        "n": {"type": "noul", "instructions": "i"},
        "c": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None, "c": None}},
        "s": {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}}}))
    ans = to_answers([[0.15, 0.35, 0.5], [0.4, 0.1, 0.1, 0.4], [0.1, 0.1, 0.4, 0.4]], meta)
    assert ans["n"]["unknown"] == 0.5 and ans["n"]["noul"] == 0.7
    assert ans["c"]["unknown"] == 0.4 and ans["c"]["probabilities"] == {"a": 0.67, "b": 0.17, "c": 0.17}
    assert ans["s"]["unknown"] == 0.4 and ans["s"]["probabilities"] == {"0": 0.17, "1": 0.17, "2": 0.67}
    for key, answer in (("c", ans["c"]), ("s", ans["s"])):
        validate_distribution(answer["probabilities"], list(answer["probabilities"]))
    assert split_unknown([0.25, 0.75], 2) == ([0.25, 0.75], None)
    with pytest.raises(ValueError, match="expected"):
        split_unknown([0.5, 0.3, 0.1, 0.1], 2)
    # with no dustbin the answers are exactly what they were before the key existed
    assert to_answers([[0.3, 0.7]], meta[:1]) == {"n": {"type": "noul", "noul": 0.7}}


def test_evidence_rides_along_when_the_request_names_sentences():
    rec, meta = to_record(SystemOneRequest.model_validate({"state": "A. B. C.", "sentences": ["A.", "B.", "C."],
                                                           "questions": {"q": {"type": "noul", "instructions": "i"}}}))
    assert rec["sentences"] == [{"text": "A."}, {"text": "B."}, {"text": "C."}]
    ans = to_answers([[0.4, 0.6]], meta, [[0.1, 0.7, 0.2]], ["A.", "B.", "C."])
    assert ans["q"]["evidence"] == [{"sentence": "B.", "p": 0.7}, {"sentence": "C.", "p": 0.2}, {"sentence": "A.", "p": 0.1}]


def test_stack_plans_are_accepted_by_the_trial_allowlist():
    from kev.experiment import validated_trial
    manifest = json.loads(Path("evals/v7/decision-v7/manifest.json").read_text())
    plan = json.loads(Path("experiments/stack-4b.json").read_text())
    assert len(plan) == 3
    trials = [validated_trial(t, manifest) for t in plan]
    assert [t["data"] for t in trials] == ["evals/stack/stack_dustbin.jsonl", "evals/stack/stack_uniform.jsonl", "evals/stack/dag_nodeps.jsonl"]
    assert trials[0]["dustbin"] == 1 and trials[0]["evidence_w"] == 0.5 and trials[0]["head_lr"] == 1e-4
    assert "dustbin" not in trials[1] and trials[1]["evidence_w"] == 0.5
    assert trials[2]["evidence_w"] == 0.0 and "dustbin" not in trials[2]
    assert all(t["replay"] == 2000 and t["init_from"] == "jaredpalmer/kev-4b" and t["lr"] == 2e-5 for t in trials)
    with pytest.raises(ValueError, match="invalid dustbin"):
        validated_trial({**plan[0], "dustbin": 2}, manifest)


def test_training_augmentation_keeps_evidence_and_deps_consistent(tok):
    """The trainer materializes augment()'s output, so the sentences must survive it (or the evidence loss silently never
    fires), and a none_pair record, being one question on its own, must not keep dependencies it cannot resolve."""
    import random
    from kev.data import augment, none_pair
    record = expand(certificate_record())
    record["questions"]["decision"]["criteria"]["escalate"] = "Send the case to a reviewer"   # 3 options: none_pair-eligible
    for seed in range(5):
        variant = augment(record, random.Random(seed), p_none=0.3, p_none_distract=0.3, p_distract=0.3)
        enc = encode(tok, materialize(variant), strict=True)
        assert len(enc["sent_idx"]) == len(record["_meta"]["sentences"])
        assert all(gold for gold in enc["evidence"]), "every expanded question has gold sentences"
        assert any(enc["deps"]), "dependencies survive augmentation"
        for single in none_pair(record, random.Random(seed)):
            enc1 = encode(tok, materialize(single), strict=True)
            assert enc1["deps"] == [()] and enc1["evidence"][0] and len(enc1["sent_idx"]) == len(record["_meta"]["sentences"])
