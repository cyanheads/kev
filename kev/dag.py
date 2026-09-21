"""Certificate -> question DAG.

A compositional policy record carries a certificate: the rule tree, its atoms, the facts, and the rendered fact order
(`kev.composition`). The label is therefore a known function of intermediate truth values that the record never asks
about. `expand` turns those intermediates into questions: one Noul per atom, one Noul per internal rule node depending
on its children, and the record's own `decision` question depending on the root's children. The question order is
atoms first, then nodes in topological order, then `decision`, so every dependency names an earlier question.

The same expansion is used by `scripts/build_stack_data.py` (to write training records) and by
`kev.benchmark --dag_expand` (in memory at predict time; frozen suites are never modified).
"""
from .composition import atom_text, atom_value, evaluate_rule, render_rule, rendered_facts

DAG_SOURCES = ("compositional", "composition_holdout")


def expandable(record):
    meta = record.get("_meta") or {}
    return bool(meta.get("certificate")) and meta.get("source") in DAG_SOURCES and "decision" in record["questions"]


def node_id(path):
    return "decision" if not path else "node_" + "_".join(str(i) for i in path)


def walk(tree, path=()):
    """(path, subtree) for every internal node of the rule tree, children before parents. The root is included last."""
    if isinstance(tree, int): return []
    out = []
    for i, child in enumerate(tree[1:]):
        out += walk(child, path + (i,))
    return out + [(path, tree)]


def child_ids(tree, path):
    """The question id answering each child of the node at `path`: an atom question for a leaf, a node question else."""
    return [f"atom_{c}" if isinstance(c, int) else node_id(path + (i,)) for i, c in enumerate(tree[1:])]


def subtree_atoms(tree):
    if isinstance(tree, int): return [tree]
    return [a for child in tree[1:] for a in subtree_atoms(child)]


def expand(record, facts=None):
    """Compositional record -> the same record with its certificate's intermediates asked as questions.

    `facts` overrides the certificate's facts (used to build evidence-removed variants): the state's rendered case is
    rebuilt from the surviving facts, and any question whose value is then UNKNOWN gets label None for the caller to
    turn into a dustbin or a uniform target. With no override every label is a bool.

    Adds `_meta["sentences"]` (one per rendered fact, carrying that fact's key) and a per-question `evidence` list of
    sentence ordinals: an atom points at the sentences rendering its fields, a node or the root at the union over its
    subtree's atoms.
    """
    cert = record["_meta"]["certificate"]
    tree, atoms = cert["tree"], cert["atoms"]
    overridden = facts is not None
    if not overridden: facts = cert["facts"]
    order = [k for k in cert["order"] if k in facts]
    style = record["_meta"].get("render_style", 0)
    sentences = [{"text": text, "facts": [key]} for text, key in zip(rendered_facts(facts, order), order)]
    position = {key: i for i, key in enumerate(order)}
    evidence_of = lambda keys: sorted(position[k] for k in keys if k in position)

    questions = {qid: q for qid, q in record["questions"].items() if qid != "decision"}
    for i, atom in enumerate(atoms):
        questions[f"atom_{i}"] = {"type": "noul", "instructions": f"Is it true that {atom_text(atom)}?",
                                 "label": atom_value(atom, facts), "src": "stack_dag_atom", "dag_role": "atom",
                                 "evidence": evidence_of(atom["fields"])}
    for path, subtree in walk(tree):
        if not path:
            continue                                  # the root is answered by the record's own `decision` question
        questions[node_id(path)] = {"type": "noul", "instructions": f"Is it true that {render_rule(subtree, atoms, style)}?",
                                    "label": evaluate_rule(subtree, atoms, facts), "src": "stack_dag_node", "dag_role": "node",
                                    "deps": child_ids(subtree, path),
                                    "evidence": evidence_of({f for a in subtree_atoms(subtree) for f in atoms[a]["fields"]})}
    root = dict(record["questions"]["decision"])
    root["deps"] = [f"atom_{tree}"] if isinstance(tree, int) else child_ids(tree, ())
    root["evidence"] = evidence_of({f for a in subtree_atoms(tree) for f in atoms[a]["fields"]})
    if overridden:
        # the frozen label belongs to the frozen facts; with an override the root is re-evaluated like every other node
        value = evaluate_rule(tree, atoms, facts)
        yes = record["questions"]["decision"]["label"] if cert["label"] else next(k for k in root["criteria"] if k != record["questions"]["decision"]["label"])
        no = next(k for k in root["criteria"] if k != yes)
        root["label"] = None if value is None else (yes if value else no)
    questions["decision"] = root

    meta = {**record["_meta"], "sentences": sentences}
    state = record["state"]
    if overridden:
        state = {**state, "case": " ".join(s["text"] for s in sentences)}
        meta = {**meta, "certificate": {**cert, "facts": facts, "order": order}}
    return {**record, "state": state, "questions": questions, "_meta": meta}


def strip_deps(record):
    """The same record with every dependency removed: the multitask control arm (same questions, isolated rows)."""
    return {**record, "questions": {qid: {k: v for k, v in q.items() if k != "deps"} for qid, q in record["questions"].items()}}
