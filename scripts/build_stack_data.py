"""Training records for the `stack` branch: question DAGs, the UNKNOWN dustbin, and the evidence pointer.

Six files under evals/stack/, each in the --data JSONL format (kev.data.load_records), plus a manifest with hashes.
Frozen: rerunning must reproduce the bytes.

  dag.jsonl               every compositional record of evals/v7/decision-v7's training partition, expanded by
                          kev.dag.expand: one Noul per rule atom, one per internal rule node depending on its
                          children, and the record's own `decision` question depending on the root's children. Each
                          record carries `_meta.sentences` (one per rendered fact) and per-question `evidence`.
  unknown_dustbin.jsonl   the same compositional records with (a) the sentence rendering the deciding fact removed, so
                          every question the missing fact decides becomes unknowable, and (b) the routing-reference
                          sentence removed, which decides nothing (the filler control). Plus freshly generated legacy
                          contrastive pairs in three variants: intact, evidence-sentence dropped, filler dropped.
                          Unknowable questions carry `unknown: true`, i.e. the dustbin key is the answer.
  unknown_uniform.jsonl   the same records with a uniform soft `target` instead of the dustbin key (the night-2 arm,
                          for a like-for-like comparison of the two ways to say "I cannot know this").
  stack_dustbin.jsonl     dag.jsonl + unknown_dustbin.jsonl
  stack_uniform.jsonl     dag.jsonl + unknown_uniform.jsonl
  dag_nodeps.jsonl        dag.jsonl with every dependency stripped: the multitask control arm (same questions, each
                          answered from the state alone).

    uv run python scripts/build_stack_data.py
"""
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev import contrastive                                          # noqa: E402
from kev.composition import evaluate_rule                            # noqa: E402
from kev.dag import expand, expandable, strip_deps                   # noqa: E402
from kev.data import load_records, materialize                       # noqa: E402
from kev.model import encode, load_tokenizer                         # noqa: E402
from kev.suite import digest, load_split, write_json                 # noqa: E402

OUT = ROOT / "evals/stack"
SEED = "stack-20260921"
SUITE = ROOT / "evals/v7/decision-v7"
# `deadline` and `authorization` are held out of training everywhere; the rest are the trainable contrastive families
LEGACY_FAMILIES = [f for f in contrastive.FAMILIES if f not in ("deadline", "authorization")]
# the Qwen3.5 tokenizer family; the 0.8B shares the vocabulary of the 4B and 9B bases
TOKENIZER = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
CONTAMINATION_SUITES = ("evals/v4/transfer-v4", "evals/v9/transfer-v9", "evals/v7/decision-v7")


def state_hash(state):
    """The hash kev.composition.generate stamps on a compositional record, so a rebuilt state is comparable with it."""
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def uniform_target(question):
    keys = list(question["criteria"]) if question["type"] == "choice" else \
        ["false", "true"] if question["type"] == "noul" else [str(i) for i in range(len(question["criteria"]))]
    return {k: 1.0 / len(keys) for k in keys}


def mark_unknown(question, intact_label, flavour):
    """A question whose evidence is gone. The label stays at what it would have been with the evidence present, so the
    record still validates; what the model is trained on is the dustbin key or the uniform target."""
    out = {**question, "label": intact_label}
    if flavour == "dustbin": out["unknown"] = True
    else: out["target"] = uniform_target(question)
    return out


# --- compositional arm -------------------------------------------------------------------------------------------

def dag_file():
    records, skipped = [], 0
    for record in load_split(SUITE, "train"):
        if record["_meta"]["source"] != "compositional" or not expandable(record):
            continue
        out = expand(record)
        out["_meta"] = {**out["_meta"], "source": "stack_dag",
                        "id": "stack_dag/" + record["_meta"]["id"], "group_id": "stack_dag/" + record["_meta"]["group_id"]}
        records.append(out)
    return records, skipped


def unknown_compositional(flavour):
    """(a) the sentence rendering the deciding fact removed, (b) the routing-reference sentence removed."""
    records, no_effect = [], 0
    for record in load_split(SUITE, "train"):
        if record["_meta"]["source"] != "compositional" or not expandable(record):
            continue
        cert = record["_meta"]["certificate"]
        intact = expand(record)
        for key, source, tag in ((cert["deciding_field"], "stack_unknown", "evidence"),
                                 ("routing reference", "stack_unknown_control", "filler")):
            if key not in cert["facts"]: continue
            facts = {k: v for k, v in cert["facts"].items() if k != key}
            out = expand(record, facts=facts)
            unknown = [qid for qid, q in out["questions"].items() if q["label"] is None]
            if tag == "evidence" and not unknown:
                no_effect += 1; continue            # the missing fact decided nothing here: not an unknowable example
            if tag == "filler" and unknown:
                raise ValueError("removing the routing reference changed a label")
            out["questions"] = {qid: (mark_unknown(q, intact["questions"][qid]["label"], flavour) if q["label"] is None else q)
                                for qid, q in out["questions"].items()}
            out["_meta"] = {**out["_meta"], "source": source, "dropped_fact": key, "unknown_questions": unknown,
                            "id": f"{source}/{record['_meta']['id']}", "group_id": f"{source}/{record['_meta']['group_id']}",
                            "text_sha256": state_hash(out["state"])}
            records.append(out)
    return records, no_effect


# --- legacy contrastive arm --------------------------------------------------------------------------------------

def legacy_variants(item, family, pair_id, sibling, order_seed, flavour):
    """One contrastive item as three records: intact, its evidence sentence dropped, a filler sentence dropped.

    `contrastive.to_request` shuffles the sentences; the same seed replays that order so `_meta.sentences` lines up
    with the rendered case (asserted below, not assumed)."""
    out = []
    evidence_at = [i for i, (_, facts) in enumerate(item["sentences"]) if facts]
    filler_at = [i for i, (_, facts) in enumerate(item["sentences"]) if not facts]
    drops = [(None, "stack_legacy", "intact"), (evidence_at[0], "stack_unknown", "evidence")]
    if filler_at: drops.append((filler_at[0], "stack_unknown_control", "filler"))
    for drop, source, tag in drops:
        kept = [s for i, s in enumerate(item["sentences"]) if i != drop]
        stripped = {**item, "sentences": kept}
        request = contrastive.to_request(stripped, family, pair_id, sibling, random.Random(order_seed))
        order = list(range(len(kept))); random.Random(order_seed).shuffle(order)
        sentences = [{"text": kept[i][0], "facts": sorted(kept[i][1])} for i in order]
        if " ".join(s["text"] for s in sentences) != request["state"]["case"]:
            raise ValueError("replayed sentence order does not match the rendered case")
        evidence = [i for i, s in enumerate(sentences) if s["facts"]]
        undetermined = contrastive.label_of(stripped) == contrastive.UNDETERMINED
        if (tag == "evidence") != undetermined:
            raise ValueError(f"{family}/{tag}: dropping this sentence left the label {'determined' if tag == 'evidence' else 'undetermined'}")
        intact = contrastive.label_of(item)
        questions = {qid: {**q, "src": f"{source}_{family}", "evidence": evidence} for qid, q in request["questions"].items()}
        request["questions"] = {qid: (mark_unknown(q, intact, flavour) if undetermined else q) for qid, q in questions.items()}
        request["_meta"].update(source=source, variant="clean", dropped_sentence=None if drop is None else item["sentences"][drop][0],
                                sentences=sentences, id=f"{source}/{family}/{pair_id}/{sibling}/{tag}",
                                group_id=f"stack_legacy/{family}/{pair_id}")
        request["_meta"].pop("pair_id", None); request["_meta"].pop("sibling", None)
        out.append(request)
    return out


def legacy_file(pairs_per_family, flavour):
    records = []
    for family in LEGACY_FAMILIES:
        rng = random.Random(f"{SEED}:legacy:{family}")
        kept, attempts = 0, 0
        while kept < pairs_per_family and attempts < 50 * pairs_per_family:
            attempts += 1
            a, b = contrastive.FAMILIES[family](rng)
            if contrastive.check_pair(a, b): continue
            pair_id = f"{SEED}-{family}-{kept:04d}"; order_seed = rng.getrandbits(64)
            for sibling, item in (("a", a), ("b", b)):
                records += legacy_variants(item, family, pair_id, sibling, order_seed, flavour)
            kept += 1
        if kept < pairs_per_family: raise ValueError(f"{family}: only {kept}/{pairs_per_family} pairs passed the checks")
    return records


# --- admission ---------------------------------------------------------------------------------------------------

def fits(tok, record):
    """Round-trip through load_records' format: materialize, then encode strictly inside the training context."""
    try:
        enc = encode(tok, materialize(record), strict=True)
    except ValueError:
        return False
    return len(enc["ids"]) <= 2048 and not enc["sentences_dropped"]


def admit(tok, records, excluded, report):
    out = []
    for r in records:
        report["considered"] += 1
        if r["_meta"].get("text_sha256") in excluded:
            report["evaluation_overlap"] += 1; continue
        if not fits(tok, r):
            report["context_rejected"] += 1; continue
        out.append(r); report["accepted"] += 1
    return out


def write(name, records, manifest):
    body = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
    (OUT / name).write_text(body)
    manifest["files"][name] = {"records": len(records), "sha256": digest(OUT / name),
                               "sources": sorted({r["_meta"]["source"] for r in records}),
                               "questions": sum(len(r["questions"]) for r in records)}
    print(f"{name}: {manifest['files'][name]['records']} records, {manifest['files'][name]['questions']} questions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_pairs", type=int, default=30, help="contrastive pairs per trainable family")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(*TOKENIZER)
    excluded = {r["_meta"]["text_sha256"] for suite in CONTAMINATION_SUITES
                for split in ("development", "test") for r in load_split(ROOT / suite, split, allow_test=True)}
    v9_unknowable = {r["_meta"]["text_sha256"] for split in ("development", "test")
                     for r in load_split(ROOT / "evals/v9/transfer-v9", split, allow_test=True)
                     if r["_meta"]["source"] in ("unknowable", "unknowable_control")}
    manifest = {"seed": SEED, "suite": str(SUITE.relative_to(ROOT)), "tokenizer": {"name": TOKENIZER[0], "revision": TOKENIZER[1]},
                "legacy_pairs_per_family": a.legacy_pairs, "legacy_families": LEGACY_FAMILIES,
                "excluded_evaluation_states": len(excluded), "files": {}, "admission": {}}

    report = Counter()
    dag, _ = dag_file()
    dag = admit(tok, dag, excluded, report)
    manifest["admission"]["dag.jsonl"] = dict(report)
    write("dag.jsonl", dag, manifest)
    write("dag_nodeps.jsonl", [strip_deps(r) for r in dag], manifest)

    for flavour in ("dustbin", "uniform"):
        report = Counter()
        composed, no_effect = unknown_compositional(flavour)
        legacy = legacy_file(a.legacy_pairs, flavour)
        # the contrastive families draw from small domains, so a fresh seed still lands on a handful of states
        # transfer-v9 already froze as unknowable items; those are evaluation states and are dropped, not trained on
        report["v9_unknowable_collisions"] = len({r["_meta"]["text_sha256"] for r in legacy} & v9_unknowable)
        records = admit(tok, composed + legacy, excluded, report)
        survivors = {r["_meta"]["text_sha256"] for r in records} & v9_unknowable
        if survivors: raise ValueError(f"{len(survivors)} written states collide with transfer-v9's unknowable records")
        report["deciding_fact_decided_nothing"] = no_effect
        manifest["admission"][f"unknown_{flavour}.jsonl"] = dict(report)
        write(f"unknown_{flavour}.jsonl", records, manifest)
        write(f"stack_{flavour}.jsonl", dag + records, manifest)
        manifest["files"][f"stack_{flavour}.jsonl"]["concat_of"] = ["dag.jsonl", f"unknown_{flavour}.jsonl"]

    write_json(OUT / "manifest.json", manifest)
    for name in manifest["files"]:
        records = load_records(OUT / name, source="stack")     # the trainer's own reader must accept every line
        print(f"{name}: load_records -> {len(records)} records, sources {sorted({r['_meta']['source'] for r in records})}", flush=True)


if __name__ == "__main__":
    main()
