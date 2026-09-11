#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — modal_train.py   (local orchestrator for the Modal GPU run)
#
#  Drives tools/modal_qat.py from this machine: chunk-loop train_chunk until
#  the step target is reached (exit 0) or the chunk budget keeps expiring
#  (exit 3 -> launch the next chunk), then report/download artifacts.
#  NOTHING is executed locally except these remote calls — checkpoints and
#  exports live in the Modal Volume; `modal volume get` fetches them home.
#
#  Commands:
#    python tools/modal_train.py status                  # Volume artifact report
#    python tools/modal_train.py qat --fresh             # round-3 calm recipe
#    python tools/modal_train.py qat --chunks 6          # resume/continue
#    python tools/modal_train.py lora --steps 2000       # assistant sidecar
#
#  Prereqs (once): pip install modal && modal token new
#  Full walkthrough: docs/MODAL_QAT_GUIDE.md
# =============================================================================
import argparse
import sys
import time

sys.path.insert(0, "tools")

import modal_qat  # noqa: E402  (local import from tools/)


def do_status() -> int:
    result = modal_qat.export_best.remote()
    print("[modal] volume artifacts (MB):")
    for name, mb in result.get("files_mb", {}).items():
        print(f"  {name:<40} {mb:>10.1f}")
    tail = result.get("qat_log_tail", "")
    if tail:
        print("[modal] qat_log tail:")
        print(tail)
    return 0


def do_qat(args) -> int:
    target = args.steps
    started = False
    for chunk in range(1, args.chunks + 1):
        print(f"[modal] QAT chunk {chunk}/{args.chunks} "
              f"(time_budget {args.time_budget}s)…", flush=True)
        t0 = time.time()
        res = modal_qat.train_chunk.remote(
            steps=target,
            time_budget=args.time_budget,
            lr=args.lr,
            kd_weight=args.kd_weight,
            window=args.window,
            batch=args.batch,
            eval_every=args.eval_every,
            fresh=(args.fresh and not started),
        )
        started = True
        print(f"[modal] chunk {chunk}: exit {res['exit']} "
              f"({time.time() - t0:.0f}s remote)")
        print(res.get("tail", "").strip())
        if res["exit"] == 0:
            print("[modal] step target reached — GGUF exported to the Volume")
            return do_status()
        if res["exit"] != 3:
            print(f"[modal] UNEXPECTED exit {res['exit']} — stopping")
            return 1
    print("[modal] chunk budget exhausted; re-run 'qat' (no --fresh) to "
          "continue from the checkpoint")
    return do_status()


def do_lora(args) -> int:
    res = modal_qat.train_lora.remote(
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
    )
    print(f"[modal] lora exit {res['exit']}")
    print(res.get("tail", "").strip())
    if res["exit"] == 0:
        print("[modal] sidecar exported to the Volume (assistant-lora.gguf)")
        return do_status()
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="report Volume artifacts + log tail")

    p = sub.add_parser("qat", help="chunked ternary QAT (round-3 defaults)")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--chunks", type=int, default=6)
    p.add_argument("--time-budget", type=int, default=3300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--fresh", action="store_true",
                   help="delete the Volume checkpoint first (round-3 clean init)")

    p = sub.add_parser("lora", help="assistant-behavior LoRA pass")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)

    args = ap.parse_args()
    if args.cmd == "status":
        return do_status()
    if args.cmd == "qat":
        return do_qat(args)
    if args.cmd == "lora":
        return do_lora(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
