# Plan: question DAGs, the UNKNOWN dustbin, and the evidence pointer

Status: **criteria fixed 2026-09-21 before the first trial ran.** Results are appended to §5 as they land; nothing above §5 changes after a trial starts.

Three additions to the pointer readout, trained together as one delta from the released Kev-4B and compared against it on the same frozen items. Code: [`kev/dag.py`](kev/dag.py), [`kev/model.py`](kev/model.py) (`rows_of`, `PointerHead(dustbin=)`, `evidence_head`), [`kev/benchmark.py`](kev/benchmark.py) (`unknown`, `evidence`, `dag`, `faithfulness` blocks). Data: [`evals/stack/`](evals/stack/manifest.json), built by [`scripts/build_stack_data.py`](scripts/build_stack_data.py) from the decision-v7 training partition only.

## 1. What is being tested

| | mechanism | why it might work | what would show it does not |
|---|---|---|---|
| **DAG** (question dependencies) | a question may name earlier questions; its row is state + those questions' branches + its own, so the root of a rule sees the atom and node questions before it decides | the compositional families are exactly the case where the answer is a known function of intermediates the model is never asked about | root accuracy on held-out rule shapes no better than the same questions asked without dependencies (arm 3) |
| **UNKNOWN dustbin** | one learned extra key in the pointer head; records whose deciding sentence was deleted are labelled with it | an explicit "the evidence does not decide this" output instead of asking the model to spread probability uniformly | `p_unknown` cannot separate transfer-v9's unknowable items from their intact controls |
| **evidence pointer** | a second pointer from `<decide>` to the token ending each sentence, trained on the sentences that carry each question's facts | atom-level questions have one or two gold sentences, so the pointer has a sharp target | deleting the top-pointed sentence moves the answer no more than deleting a random one |

## 2. Arms

All three are deltas from `jaredpalmer/kev-4b` (lr 2e-5, one epoch, 2000 replay records from decision-v7, bf16, batch 4 × accum 2, seed 1), [`experiments/stack-4b.json`](experiments/stack-4b.json):

| arm | data | flags | answers |
|---|---|---|---|
| 1 `stack_dustbin` | dag + unknown records with the dustbin label | `--dustbin 1 --evidence_w 0.5 --head_lr 1e-4` | the stack as designed |
| 2 `stack_uniform` | the same records, unknown questions carry a uniform soft target instead | `--evidence_w 0.5 --head_lr 1e-4` | is the dustbin better than the existing way of saying "I cannot know" |
| 3 `dag_nodeps` | the dag records with every dependency stripped | none | is it the dependency edge or just the extra questions (multitask control) |

Arm 1 runs first on a local RTX 3090; arms 2 and 3 run only if arm 1 clears its kill criteria. Arm 3 is the one baseline that cannot be skipped: without it a DAG win is not attributable to the edges.

## 3. Reference row (released Kev-4B, transfer-v9 development, [`runs/kev-4b-transfer-v9`](runs/kev-4b-transfer-v9/00-kev-4b/result.json))

| knowable acc (n=1046) | held-out rules: and_or / conditional / or_not (n=32 each) | deadline | unknowable mean max-p / share ≥ 0.9 | control share ≥ 0.9 | cov@5 % err |
|---|---|---|---|---|---|
| 0.729 | 0.781 / 0.875 / 0.969 | 0.525 | 0.779 / 0.436 | 0.900 | 0.251 |

## 4. Criteria, fixed in advance

Every number is on transfer-v9 development (transfer-v4 is a byte-identical subset) with `--dag_expand --faithfulness`; the record's own question is scored exactly as before, the added atom and node questions only appear in the `dag` block.

**Retention (applies to every arm).** Knowable accuracy within 2 points of 0.729 and MMLU / PAWS within 3 points of the reference; the delta recipe held this on the night-2 runs, so a miss here is a bug, not a finding.

**DAG.** Pass: mean held-out-rule root accuracy ≥ the reference (0.875) *and* atom accuracy ≥ 0.90 in the `dag` block. Attributable: arm 1 root accuracy exceeds arm 3 by more than the paired bootstrap CI. Kill: root accuracy below the reference by more than 3 points. Breakthrough (the bar set before this started): ≥ 0.90 on every held-out rule family with a CI clear of the reference.

**Dustbin.** Pass: AUROC of `p_unknown` (unknowable vs intact control, 110 pairs) ≥ 0.90 and `unknowable_share_at_0_9_combined` ≤ 0.10 with the control share staying ≥ 0.80. Compare with arm 2 on the existing `unknowable.share_at_0_9` (the uniform arm has no `p_unknown`). Kill: AUROC < 0.75.

**Evidence.** Scored on the atom questions of the held-out rule records (`evidence.stack_dag_atom`, 288 questions): each atom has one gold sentence among 4–6, so chance is about 0.2. The root's gold set is nearly every sentence, so its top-1 is close to 1 for any pointer and is not a criterion. Pass: atom top-1 ≥ 0.60. Faithful: the `faithfulness` paired confidence-drop difference has a 95 % CI above zero on at least two of the three families. Kill: atom top-1 below 0.35, or every family's CI including zero once arm 1 has passed everything else.

**Cost.** The whole programme stays under the $30 Modal budget; arm 1 costs nothing (3090). A second seed of any arm is bought only for a result that decides something.

Noise floor: 96 held-out rule records and 110 unknowable pairs; a 5-point difference on the rule families is inside the bootstrap CI, so single-seed differences below that are reported as ties.

## 5. Results

_(appended as they land)_
