"""j-lens and r-lens: fitted Jacobian lenses for Llama-3.1-8B, Qwen2.5-7B, Qwen3-4B.

Subcommands (run `python lens.py <cmd> --help`):
  make-data  Regenerate the committed fitting corpus and eval prompt sets (seeded).
             -> data/prompts.txt, data/eval_prompts.json
  fit        Fit a lens: average Jacobian of the final residual w.r.t. each layer's residual
             over the corpus. --lens r runs the identical loop under the LRP rules context.
             Resume-safe (atomic checkpoints). (~1.5-5 h per lens at --n-prompts 100)
             -> results/{model}/{j,r}lens.pt, ckpt_{j,r}.pt, fit_{j,r}lens.json
  apply      Per-layer top-k readout of a prompt through chosen lenses (stdout).
  eval       Multihop top-5 tables + top-1 agreement curves, all three readouts. (~10 min)
             -> results/{model}/eval_multihop.json, agreement.json, grid_boot.png, agreement.png
  sanity     Verification suite on tiny fits: final-layer identity, readout==model logits,
             merge, r-rules forward invariance, resume, boot-prompt smoke. (~15-30 min)
             -> results/{model}/sanity.json
  merge      n_prompts-weighted merge of lenses fit on disjoint prompt slices.

j-lens is a from-scratch reimplementation of the Jacobian lens from
https://github.com/anthropics/jacobian-lens (Apache-2.0; reference only, not vendored):
J_l = E[dh_final/dh_l] averaged over prompts and positions, readout unembed(J_l @ h).
r-lens follows "R-lens: making J-lens more faithful on early layers"
(https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/): same fit, but the backward pass
propagates LRP coefficients (detached RMSNorm denominator, SiLU identity rule, half-rule on
the SwiGLU gate), which stops gradient error accumulating across layers and makes early-layer
readouts far more interpretable. The logit lens falls out as the transport-free baseline.
"""
import argparse
import contextlib
import time

import torch

from lib.evals import MULTIHOP, agreement, readout_modes
from lib.fitting import DIM_BATCH, MAX_SEQ_LEN, fit
from lib.layout import EMBED, MODELS, load
from lib.lens import Lens, topk_tokens
from lib.plotting import agreement_curves, token_grid
from lib.prompts import make_agreement_prompts, make_prompts
from lib.util import DATA, load_json, results_dir, save_json

BOOT = MULTIHOP[0]  # the boot/currency multihop prompt from the upstream README


def corpus():
    return (DATA / "prompts.txt").read_text().splitlines()


def rules_ctx(layout, kind):
    from lib.rules import rlens_rules
    return rlens_rules(layout) if kind == "r" else contextlib.nullcontext()


def identity_err(lens, layout):
    J = lens.jacobians[layout.n_layers - 1].float()
    return (J - torch.eye(lens.d_model)).abs().max().item()


# ==== make-data ==================================================================================

def cmd_make_data(args):
    DATA.mkdir(exist_ok=True)
    prompts = make_prompts(n=300, seed=0)
    (DATA / "prompts.txt").write_text("\n".join(prompts) + "\n")
    print(f"wrote {DATA / 'prompts.txt'} ({len(prompts)} prompts)")
    save_json(DATA / "eval_prompts.json",
              {"multihop": MULTIHOP, "agreement": make_agreement_prompts(n=50, seed=1)})
    print(f"wrote {DATA / 'eval_prompts.json'}")


# ==== fit ========================================================================================

def cmd_fit(args):
    out = results_dir(args.model)
    layout = load(args.model)
    prompts = corpus()[:args.n_prompts]
    t0 = time.time()
    with rules_ctx(layout, args.lens):
        lens = fit(layout, prompts, kind=args.lens, dim_batch=args.dim_batch,
                   max_seq_len=args.max_seq_len, checkpoint_path=out / f"ckpt_{args.lens}.pt",
                   resume=args.resume)
    lens.save(out / f"{args.lens}lens.pt")
    save_json(out / f"fit_{args.lens}lens.json",
              dict(lens.config, n_prompts_used=lens.n_prompts, wall_s=round(time.time() - t0),
                   final_layer_identity_err=identity_err(lens, layout)))
    print(f"fit done: {lens.n_prompts} prompts, "
          f"final-layer identity err {identity_err(lens, layout):.2e}")


# ==== apply ======================================================================================

def cmd_apply(args):
    layout = load(args.model)
    out = results_dir(args.model)
    for kind in args.lens.split(","):
        if kind == "logit":
            lens = Lens.load(out / "jlens.pt") if (out / "jlens.pt").exists() else \
                Lens.load(out / "rlens.pt")
        else:
            lens = Lens.load(out / f"{kind}lens.pt")
        lens_logits, model_logits, _ = lens.apply(
            layout, args.prompt, positions=(args.position,), use_jacobian=(kind != "logit"))
        print(f"\n==== {kind} readout, position {args.position} ====")
        for l in sorted(lens_logits):
            toks = topk_tokens(layout.tok, lens_logits[l], args.topk)[0]
            lab = "emb" if l == EMBED else f"L{l:>3}"
            print(f"  {lab}: " + "  ".join(f"{t!r}({v:+.1f})" for t, v in toks))
        print("  model out: "
              + "  ".join(f"{t!r}({v:+.1f})" for t, v in
                          topk_tokens(layout.tok, model_logits, args.topk)[0]))


# ==== eval =======================================================================================

def cmd_eval(args):
    out = results_dir(args.model)
    layout = load(args.model)
    lenses = {k: Lens.load(out / f"{k}lens.pt").to("cuda")
              for k in ("j", "r") if (out / f"{k}lens.pt").exists()}
    assert lenses, f"no fitted lenses in {out}"
    ev = load_json(DATA / "eval_prompts.json")

    tables = {p: readout_modes(layout, lenses, p) for p in ev["multihop"]}
    save_json(out / "eval_multihop.json", tables)
    token_grid(tables[BOOT], out / "grid_boot.png",
               f"{args.model}: {BOOT!r} (pos -2)")

    agree = agreement(layout, lenses, ev["agreement"])
    save_json(out / "agreement.json", agree)
    agreement_curves(agree, out / "agreement.png",
                     f"{args.model}: top-1 agreement with final output (50 held-out prompts)")
    for m, per_layer in agree.items():
        early = [per_layer[l] for l in sorted(per_layer)[:len(per_layer) // 2]]
        print(f"{m:>6}: early-half mean agreement {sum(early) / len(early):.3f}")


# ==== sanity =====================================================================================

def cmd_sanity(args):
    out = results_dir(args.model)
    layout = load(args.model)
    prompts = corpus()[:4]
    res = {}

    # r-rules forward invariance (bit-identical logits)
    ids = layout.encode(BOOT)
    with torch.no_grad():
        base = layout.model(input_ids=ids).logits
        with rules_ctx(layout, "r"):
            ruled = layout.model(input_ids=ids).logits
    res["rules_forward_bit_identical"] = bool(torch.equal(base, ruled))

    # tiny fits: j on 4 prompts (also halves: 2+2 for the merge check), r on 2 prompts
    kw = dict(max_seq_len=64)
    lens4 = fit(layout, prompts, kind="j", **kw)
    lens_a = fit(layout, prompts[:2], kind="j", **kw)
    lens_b = fit(layout, prompts[2:], kind="j", **kw)
    with rules_ctx(layout, "r"):
        rlens2 = fit(layout, prompts[:2], kind="r", **kw)

    # final-layer identity (both lens kinds: nothing sits between h_last and h_last)
    res["j_final_layer_identity_err"] = identity_err(lens4, layout)
    res["r_final_layer_identity_err"] = identity_err(rlens2, layout)

    # final-layer readout == model logits (validates the unembed path, incl. tied embeddings)
    lens_logits, model_logits, _ = lens4.apply(layout, BOOT, positions=(-2,))
    err = (lens_logits[layout.n_layers - 1] - model_logits).abs().max().item()
    res["final_layer_readout_max_abs_err"] = err

    # merge of disjoint halves == joint fit
    merged = Lens.merge([lens_a, lens_b])
    res["merge_max_abs_err"] = max(
        (merged.jacobians[l] - lens4.jacobians[l]).abs().max().item()
        for l in lens4.source_layers)

    # resume: interrupt after 2 prompts, resume to 4, compare against the uninterrupted fit
    ckpt = out / "ckpt_sanity.pt"
    ckpt.unlink(missing_ok=True)
    fit(layout, prompts[:2], kind="j", checkpoint_path=ckpt, checkpoint_every=1, **kw)
    resumed = fit(layout, prompts, kind="j", checkpoint_path=ckpt, resume=True, **kw)
    res["resume_max_abs_err"] = max(
        (resumed.jacobians[l] - lens4.jacobians[l]).abs().max().item()
        for l in lens4.source_layers)
    ckpt.unlink(missing_ok=True)

    # qualitative smoke: tiny-lens boot readout at a few layers (recorded, not asserted)
    mid = layout.n_layers // 2
    res["boot_smoke"] = {str(l): topk_tokens(layout.tok, lens_logits[l], 5)[0]
                         for l in [mid, mid + layout.n_layers // 4, layout.n_layers - 1]}

    res["pass"] = (res["rules_forward_bit_identical"]
                   and res["j_final_layer_identity_err"] < 1e-3
                   and res["r_final_layer_identity_err"] < 1e-3
                   and res["final_layer_readout_max_abs_err"] < 1e-2
                   and res["merge_max_abs_err"] < 1e-5
                   and res["resume_max_abs_err"] == 0.0)
    save_json(out / "sanity.json", res)
    for k, v in res.items():
        if k != "boot_smoke":
            print(f"  {k}: {v}")
    print("SANITY", "PASS" if res["pass"] else "FAIL")


# ==== merge ======================================================================================

def cmd_merge(args):
    merged = Lens.merge([Lens.load(p) for p in args.inputs])
    merged.save(args.out)


# ==== CLI ========================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-data", help="regenerate data/ (seeded, deterministic)")
    p.set_defaults(fn=cmd_make_data)

    p = sub.add_parser("fit", help="fit a lens over the corpus")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--lens", required=True, choices=["j", "r"])
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--dim-batch", type=int, default=DIM_BATCH)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("apply", help="per-layer top-k readout of a prompt")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--prompt", default=BOOT)
    p.add_argument("--lens", default="logit,j,r", help="comma list of logit,j,r")
    p.add_argument("--position", type=int, default=-2)
    p.add_argument("--topk", type=int, default=5)
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("eval", help="multihop tables + agreement curves")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("sanity", help="verification suite on tiny fits")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.set_defaults(fn=cmd_sanity)

    p = sub.add_parser("merge", help="merge lenses fit on disjoint prompt slices")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)
