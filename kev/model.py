"""Decision model: causal LM backbone + block-causal branch mask + pointer readout."""
import math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt, /opt, decide) so no
# embedding rows need to be added/trained; LoRA adapts their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH = 384, 1024


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: instruction/state tokens, and the <decide> token


def sentence_token_indices(tok, text, sentences):
    """Index, within the state's own token list, of the token holding each sentence's last character.

    Offsets are computed on the text `user_tokens` actually feeds the tokenizer (`<|x|>` rewritten to `<¦x¦>`, same
    character count), so the mapping matches the ids in the encoding. Sentences are located verbatim, in order.
    """
    rewritten = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    offsets = tok(rewritten, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out, cursor = [], 0
    for s in sentences:
        needle = _SPECIAL_RE.sub(r"<¦\1¦>", s["text"])
        at = rewritten.find(needle, cursor)
        if at < 0:
            raise ValueError(f"sentence not found in the rendered state: {s['text']!r}")
        cursor = at + len(needle)
        last = cursor - 1
        token = next((j for j, (a, b) in enumerate(offsets) if a <= last < b), None)
        if token is None:
            raise ValueError(f"no token covers the end of sentence {s['text']!r}")
        out.append(token)
    return out


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (index of </opt> token for each option), opt (per-token option index within its
    question: OPT_NONE for state/instruction, 0..K-1 for option spans, OPT_DECIDE for <decide>).

    option_isolation=True: every option span is its own sub-branch (it sees state + instruction + itself only), all
    option spans share the same position ids, and <decide> sits at one fixed position after the longest span. Then the
    per-option representations and <decide>'s attention over them are permutation-invariant by construction.

    Question DAGs: a question may carry `deps` (indices of earlier questions of the same record). The packing is
    unchanged - deps only change the row form (rows_of), where a question's row also carries its dependency closure.

    Evidence pointer: `rec["sentences"]` ([{"text", "facts"}] in rendered order) yields `sent_idx`, the state-token
    index of each sentence's last token. Sentences pushed past `max_state` are dropped (counted in
    `sentences_dropped`), and `evidence` maps each question's gold sentence ordinals (`q["evidence"]`) to positions in
    `sent_idx`, or None when the question has no gold or one of its positives was dropped.
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx, deps = [], [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ValueError(f"branch too long: {len(br)}")
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0 + len(instr))) + [p0 + len(instr) + i for sp in spans for i in range(len(sp))] + [p0 + len(instr) + longest]
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
        q_deps = tuple(q.get("deps") or ())
        if any(not 0 <= d < k - 1 for d in q_deps):
            raise ValueError(f"question {k - 1} depends on {sorted(q_deps)}; dependencies must be earlier questions")
        deps.append(q_deps)
    if option_isolation and any(deps):
        raise ValueError("option_isolation needs the packed mask; question dependencies need the row form")
    sent_idx, kept, dropped = [], {}, 0
    for i, j in enumerate(sentence_token_indices(tok, rec["state"], rec["sentences"]) if rec.get("sentences") else []):
        if 1 + j < len(S): kept[i] = len(sent_idx); sent_idx.append(1 + j)
        else: dropped += 1
    evidence = [None if not (gold := q.get("evidence")) or any(s not in kept for s in gold) else [kept[s] for s in gold]
                for q in rec["questions"]]
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "deps": deps, "sent_idx": sent_idx, "sentences_dropped": dropped, "evidence": evidence,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + 1 > max_state}


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence.

    Padded key positions are masked for every query; padded query rows keep the diagonal so no row is fully
    masked (finfo.min, not -inf, so softmax stays finite either way). Real tokens never see pads because pads sit
    after them (causal) and belong to no segment (-1).

    opts (option isolation): within a question, an option-span token may attend to state, the instruction, and its own
    span only; <decide> attends to everything in its question. Instruction tokens never see option spans (causal)."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), L), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def _softmax_or_none(logits):
    """Evidence pointer logits -> probabilities, or None when the checkpoint or the record has no sentences."""
    if not logits or all(z is None for z in logits): return None
    return [None if z is None else F.softmax(z, -1).cpu() for z in logits]


def dependency_closure(deps, k):
    """Every question question k transitively depends on, ascending. Dependencies point backwards only, so ascending
    index order is a topological order."""
    seen, stack = set(), list(deps[k])
    while stack:
        j = stack.pop()
        if j in seen: continue
        seen.add(j); stack.extend(deps[j])
    return sorted(seen)


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide", "opts"}: the branch tokens of question
    k with their (already state-continuing) positions, and the readout offsets *within the branch*. Feeding
    state + rows[k] as one causal row is equivalent to the packed block-causal form for that question, on any
    architecture: the row contains exactly the tokens question k may attend to, in the same positions.

    A question with `deps` composes its row from the branches of its whole dependency closure (topological order)
    followed by its own branch, with positions running sequentially from len(state) through the row. A question
    without deps produces exactly the packed slice, positions included."""
    seg = enc["seg"]; Ls = seg.count(0)
    spans, start = [], Ls
    for k, d in enumerate(enc["decide_idx"], start=1):
        end = d + 1                                    # <decide> is the last token of its branch
        if seg[start] != k or seg[end - 1] != k: raise ValueError("branch layout mismatch")
        spans.append((start, end)); start = end
    deps = enc.get("deps") or [()] * len(spans)
    rows = []
    for k, ((start, end), d, oi) in enumerate(zip(spans, enc["decide_idx"], enc["opt_idx"])):
        closure = dependency_closure(deps, k)
        if not closure:
            rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start, "opts": [o - start for o in oi]})
            continue
        prefix = [t for j in closure for t in enc["ids"][spans[j][0]:spans[j][1]]]
        row = prefix + enc["ids"][start:end]
        rows.append({"ids": row, "pos": list(range(Ls, Ls + len(row))), "decide": len(prefix) + d - start,
                     "opts": [len(prefix) + o - start for o in oi]})
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


class PointerHead(nn.Module):
    def __init__(self, d, dp=256, dustbin=False):
        """dp = pointer dimension (head capacity knob).

        dustbin=True appends one learned UNKNOWN key, so the readout is a distribution over K+1 outcomes: the K options
        plus "none of these is supported by the evidence". `null_bias` is added after the 1/sqrt(dp) scale and starts
        at -5, so the dustbin logit really is -5 at init (quiet next to a warm-started head's option logits)."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.dustbin = dustbin
        if dustbin:
            self.null_key = nn.Parameter(torch.zeros(dp))
            self.null_bias = nn.Parameter(torch.tensor(-5.0))

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K] (or [K+1] with a dustbin)
        q = self.q(h_decide)
        z = (self.k(h_opts) @ q) * self.scale
        if self.dustbin:
            z = torch.cat([z, ((self.null_key @ q) * self.scale + self.null_bias).reshape(1)])
        return z


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False, special_embeddings=False, lora_targets="all", dtype=torch.float32,
                 dustbin=False, evidence=False):
        super().__init__()
        # backbone only (no vocab head): we never generate text.
        # eager on MPS/CPU (known-good with our float 4D mask); SDPA on CUDA (accepts arbitrary additive masks).
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        # dtype: fp32 for training and exact evaluation; bf16 is a serving option for large backbones (8B on a 32 GB Mac)
        self.lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn).model
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        # hybrid backbones (Qwen3.5: Gated DeltaNet layers, recurrent) cannot honour the block-causal mask, so every
        # question runs as its own causal row continuing from the state (rows_of). Attention-only backbones keep the
        # packed form; the two agree to fp32 noise (tests/test_v3.py::test_rows_match_packed).
        cfg = self.lm.config
        self.hybrid = "linear_attention" in set(getattr(cfg, "layer_types", None) or [])
        if self.hybrid and option_isolation: raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
        self.option_isolation = option_isolation
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = {"trainable_token_indices": {"embed_tokens": [tok.convert_tokens_to_ids(t) for t in SPECIAL]}} if special_embeddings else {}
            targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # "all" minus the DeltaNet projections on hybrids (retention ablation)
                       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}[lora_targets]
            if self.hybrid and lora_targets in ("all", "attn"):
                # Gated DeltaNet projections (transformers 5 names, verified on Qwen3_5TextModel); the mixer's out_proj too
                targets = targets + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets, **extra)
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim, dustbin=dustbin)
        # second pointer: same <decide> query, keys are the state tokens that end each sentence (kev.model.encode)
        self.evidence_head = PointerHead(self.lm.config.hidden_size, dp=head_dim) if evidence else None
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        """encode() with this model's option-isolation setting; use this from serving/eval code."""
        return encode(tok, rec, option_isolation=self.option_isolation, **kw)

    def head_parameters(self):
        """Every pointer-head parameter (answer head incl. dustbin, evidence head): what --head_lr governs."""
        return list(self.head.parameters()) + (list(self.evidence_head.parameters()) if self.evidence_head is not None else [])

    def needs_rows(self, encs):
        """Recurrent backbones cannot honour the packed block-causal mask; neither can a question that must see another
        question's branch. Both are served by the row form."""
        return self.hybrid or any(any(d) for e in encs for d in (e.get("deps") or ()))

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    SHAPE_BUCKET = int(__import__("os").environ.get("KEV_SHAPE_BUCKET", "64"))   # MPS: pad the sequence to a multiple of this (per-shape kernel warm-up); 1 disables

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records. Pads are masked keys and sit after every
        real token, so padding never changes a real token's hidden state (parity measured exact)."""
        L = max(len(e["ids"]) for e in encs)
        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(encs), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(encs), L), dtype=torch.long, device=self.device)
        for b, e in enumerate(encs):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"], device=self.device)
            pos[b, : len(e["pos"])] = torch.tensor(e["pos"], device=self.device)
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        lm_dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=lm_dtype, opts=[e["opt"] for e in encs] if isolate else None, length=L)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()   # head stays fp32

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def _evidence_logits(self, h, decide_positions, enc):
        """Pointer logits over this record's sentence keys, one list per position in `decide_positions`. `h` must hold
        the state tokens at enc["sent_idx"] (the packed pass and every row both start with the state; the serving
        prefix path passes the cached state hidden states instead). None when there is nothing to point at."""
        if self.evidence_head is None or not enc.get("sent_idx"): return None
        keys = h[torch.tensor(enc["sent_idx"], device=self.device)]
        return [self.evidence_head(h[d], keys) for d in decide_positions]

    def forward_rows_batch(self, encs, evidence=False):
        """Row form: every question of every record is one causal row = state tokens + its branch tokens, right-padded
        into a single batch. Returns the same nested logits as forward_batch. Exact isolation by construction (rows are
        independent); the state is recomputed per row (Q x state tokens), which training accepts; serving uses the
        prefix cache instead."""
        rows, owners = [], []
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"], len(S) + r["decide"], [len(S) + o for o in r["opts"]])); owners.append(b)
        L = max(len(ids) for ids, *_ in rows)
        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos, _, _) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device); pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device); att[i, : len(rid)] = 1
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att).last_hidden_state.float()
        out, ev = [[] for _ in encs], [[] for _ in encs]
        for i, (b, (_, _, d, oi)) in enumerate(zip(owners, rows)):
            out[b].append(self.head(h[i, d], h[i, torch.tensor(oi, device=self.device)]))
            if evidence:
                pointed = self._evidence_logits(h[i], [d], encs[b])
                ev[b].append(pointed[0] if pointed else None)
        return (out, ev) if evidence else out

    def forward(self, enc, evidence=False):
        """Returns list of logits tensors, one per question (and, with evidence=True, the sentence-pointer logits)."""
        if self.needs_rows([enc]):
            out = self.forward_rows_batch([enc], evidence=evidence)
            return (out[0][0], out[1][0]) if evidence else out[0]
        h = self.hidden(enc)
        z = self._readout(h, enc)
        return (z, self._evidence_logits(h, enc["decide_idx"], enc)) if evidence else z

    def forward_batch(self, encs, evidence=False):
        """List (per record) of lists (per question) of logits, from one padded forward pass."""
        if self.needs_rows(encs): return self.forward_rows_batch(encs, evidence=evidence)
        hs = self.hidden_batch(encs)
        out = [self._readout(hs[b], e) for b, e in enumerate(encs)]
        return (out, [self._evidence_logits(hs[b], e["decide_idx"], e) for b, e in enumerate(encs)]) if evidence else out

    @torch.no_grad()
    def probs(self, enc, evidence=False):
        if not evidence: return [F.softmax(z, -1).cpu() for z in self.forward(enc)]
        zs, ez = self.forward(enc, evidence=True)
        return [F.softmax(z, -1).cpu() for z in zs], _softmax_or_none(ez)

    # --- state-prefix reuse (serving): the state is encoded once, question branches attend to its cached keys/values.
    # Exact by construction: branch tokens never attend to each other across questions (block-causal mask) and the state
    # never sees the branches (causal), so the state's hidden states and KV are identical with or without the branches.

    def _branch_rows_from_prefix(self, enc, cache, h_state=None, evidence=False):
        """Row serving: replicate the cached state once per question and run the branches as causal rows (exactly the
        forward_rows_batch layout, minus the recomputed state). The cache is consumed (replicated, then extended).
        Used for hybrid backbones and for records whose questions carry dependencies."""
        S, Sp, rows = rows_of(enc); Q = len(rows)
        cache.reorder_cache(torch.zeros(Q, dtype=torch.long, device=self.device))
        W = max(len(r["ids"]) for r in rows)
        if str(self.device) == "mps": W = -(-W // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((Q, W), self.pad_id, device=self.device); pos = torch.zeros((Q, W), dtype=torch.long, device=self.device)
        att = torch.zeros((Q, len(S) + W), dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            ids[i, : len(r["ids"])] = torch.tensor(r["ids"], device=self.device); pos[i, : len(r["pos"])] = torch.tensor(r["pos"], device=self.device); att[i, : len(S) + len(r["ids"])] = 1
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state.float()
        ps = [F.softmax(self.head(h[i, r["decide"]], h[i, torch.tensor(r["opts"], device=self.device)]), -1).cpu() for i, r in enumerate(rows)]
        if not evidence: return ps
        # the state is not recomputed here, so the sentence keys come from the cached state hidden states
        ez = None if self.evidence_head is None or not enc.get("sent_idx") else \
            [self.evidence_head(h[i, r["decide"]], h_state[torch.tensor(enc["sent_idx"], device=self.device)]) for i, r in enumerate(rows)]
        return ps, _softmax_or_none(ez)

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only. Returns (n_state_tokens, kv cache, state hidden states [Ls, d])."""
        from transformers import DynamicCache
        Ls = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device); pos = torch.tensor([enc["pos"][:Ls]], device=self.device)
        # the cache must know the layer types (hybrid backbones keep recurrent + conv states per DeltaNet layer)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        return Ls, out.past_key_values, out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc, evidence=False):
        """One full pass that also returns the state prefix (KV cropped to the state, state hidden states): a cache miss
        costs a single forward pass, not two."""
        from transformers import DynamicCache
        Ls = enc["seg"].count(0)
        if self.needs_rows([enc]):
            # recurrent layers cannot be cropped back to the state, so a hybrid miss is state pass + branch rows (the
            # state pass is kept as the reusable prefix by running it twice? no: copy the cache before consuming it)
            Ls, cache, h_state = self.prefix(enc)
            import copy
            return self._branch_rows_from_prefix(enc, copy.deepcopy(cache), h_state, evidence), (Ls, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device); pos = torch.tensor([enc["pos"]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        h = out.last_hidden_state[0].float()
        out.past_key_values.crop(-(len(enc["ids"]) - Ls))     # keep the state only (negative = drop that many trailing tokens; positive form deprecated in transformers 5)
        ps = [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]
        result = (ps, _softmax_or_none(self._evidence_logits(h, enc["decide_idx"], enc))) if evidence else ps
        return result, (Ls, out.past_key_values, h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix, evidence=False):
        """probs() for a record whose state tokens equal the cached prefix's; only the branches run. The cache is cropped
        back to the state afterwards so it can be reused."""
        Ls, cache, h_state = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        if self.needs_rows([enc]):
            import copy
            return self._branch_rows_from_prefix(enc, copy.deepcopy(cache), h_state, evidence)   # the stored prefix stays pristine
        ids = torch.tensor([enc["ids"][Ls:]], device=self.device); pos = torch.tensor([enc["pos"][Ls:]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)[:, :, Ls:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - Ls))
        ps = [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]
        return (ps, _softmax_or_none(self._evidence_logits(h, enc["decide_idx"], enc))) if evidence else ps

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
