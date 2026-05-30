#!/usr/bin/env python3
"""Train CLRS bridge with validation-gated curriculum progression.

This is a thin orchestrator around
``train_clrs_text_bridge_benchmark_native.py``. It calls the isolated trainer
one epoch at a time, reads the metrics JSON, and advances from level N to N+1
only when the chosen validation metric reaches the configured threshold.
"""

from __future__ import annotations

import argparse
import json
import numbers
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_thresholds(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition(":")
        if not value:
            raise ValueError(f"Invalid threshold item: {item!r}")
        out[int(key)] = float(value)
    return out


def _metric_from_history(metrics_path: Path, metric_path: str) -> float:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not history:
        return float("-inf")
    current: Any = history[-1]
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric path {metric_path!r} missing at {part!r}")
        current = current[part]
    return float(current)


def _latest_history_row(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if not history:
        return {}
    row = history[-1]
    return row if isinstance(row, dict) else {}


def _wandb_log_from_history(row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    log_payload: dict[str, Any] = {
        "global_epoch": int(event["global_epoch"]),
        "curriculum/level": int(event["level"]),
        "curriculum/local_epoch": int(event["local_epoch"]),
        "curriculum/threshold": float(event["threshold"]),
        "curriculum/advance_metric_value": float(event["metric_value"]),
        "curriculum/advanced": int(bool(event["advanced"])),
    }
    if isinstance(row.get("lr"), numbers.Number):
        log_payload["lr"] = float(row["lr"])
    for split in ("train", "val"):
        metrics = row.get(split, {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                log_payload[f"{split}/{key}"] = float(value)
    return log_payload


def _copy_args(args: argparse.Namespace) -> list[str]:
    passthrough = [
        "--arch",
        args.arch,
        "--llm-model-name",
        args.llm_model_name,
        "--llm-layer-index",
        str(args.llm_layer_index),
        "--llm-dtype",
        args.llm_dtype,
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-source-length",
        str(args.max_source_length),
        "--max-nodes",
        str(args.max_nodes),
        "--bridge-dim",
        str(args.bridge_dim),
        "--bridge-heads",
        str(args.bridge_heads),
        "--bridge-layers",
        str(args.bridge_layers),
        "--bridge-dropout",
        str(args.bridge_dropout),
        "--query-pos-dropout",
        str(args.query_pos_dropout),
        "--count-blend-alpha",
        str(args.count_blend_alpha),
        "--count-decode-mode",
        args.count_decode_mode,
        "--scalar-head-type",
        args.scalar_head_type,
        "--scalar-head-hidden-mult",
        str(args.scalar_head_hidden_mult),
        "--edge-text-prior-logit-scale",
        str(args.edge_text_prior_logit_scale),
        "--transnar-recurrence-steps",
        str(args.transnar_recurrence_steps),
        "--feature-quant-bins",
        str(args.feature_quant_bins),
        "--continuous-feature-clip",
        str(args.continuous_feature_clip),
        "--seed",
        str(args.seed),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--present-threshold",
        str(args.present_threshold),
        "--scalar-exact-tolerance",
        str(args.scalar_exact_tolerance),
        "--checkpoint-every-epochs",
        str(args.checkpoint_every_epochs),
        "--log-every-batches",
        str(args.log_every_batches),
        "--wandb-project",
        args.wandb_project,
        "--count-loss-weight-implicit",
        str(args.count_loss_weight_implicit),
        "--count-loss-weight-explicit",
        str(args.count_loss_weight_explicit),
        "--count-loss-weight-blended",
        str(args.count_loss_weight_blended),
        "--loss-weight-present",
        str(args.loss_weight_present),
        "--loss-weight-count",
        str(args.loss_weight_count),
        "--loss-weight-node-scalar",
        str(args.loss_weight_node_scalar),
        "--loss-weight-node-discrete",
        str(args.loss_weight_node_discrete),
        "--loss-weight-edge-scalar",
        str(args.loss_weight_edge_scalar),
        "--loss-weight-edge-discrete",
        str(args.loss_weight_edge_discrete),
        "--edge-binary-pos-weight-cap",
        str(args.edge_binary_pos_weight_cap),
        "--edge-binary-neg-penalty",
        str(args.edge_binary_neg_penalty),
        "--loss-weight-graph-scalar",
        str(args.loss_weight_graph_scalar),
        "--loss-weight-graph-discrete",
        str(args.loss_weight_graph_discrete),
        "--loss-weight-present-monotonic",
        str(args.loss_weight_present_monotonic),
        "--max-train-rows",
        str(args.max_train_rows),
        "--max-val-rows",
        str(args.max_val_rows),
        "--no-wandb",
    ]
    if args.qformer_edge_self_bias:
        passthrough.append("--qformer-edge-self-bias")
    if args.qformer_rel_pos_self_bias:
        passthrough.append("--qformer-rel-pos-self-bias")
    return passthrough


def _level_hidden_cache_args(args: argparse.Namespace, level: int) -> list[str]:
    if not args.hidden_cache_dir:
        return []
    cache_dir = Path(args.hidden_cache_dir).resolve() / f"level_{level}"
    train_cache = cache_dir / "train.pt"
    val_cache = cache_dir / "val.pt"
    if not train_cache.exists() or not val_cache.exists():
        raise FileNotFoundError(
            f"Missing hidden-state cache for level {level}: {train_cache} and {val_cache}. "
            "Build caches first or omit --hidden-cache-dir."
        )
    return ["--train-hidden-cache", str(train_cache), "--val-hidden-cache", str(val_cache)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic validation-gated CLRS bridge curriculum")
    parser.add_argument("--curriculum-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--levels", default="0,1,2,3,4")
    parser.add_argument("--advance-metric", default="val.selection_score")
    parser.add_argument("--advance-thresholds", default="0:0.80,1:0.75,2:0.70,3:0.65,4:0.60")
    parser.add_argument("--max-epochs-per-level", type=int, default=12)
    parser.add_argument("--min-epochs-per-level", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--trainer", default="")
    parser.add_argument("--hidden-cache-dir", default="")

    parser.add_argument("--arch", choices=["qformer", "transnar"], default="qformer")
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--llm-layer-index", type=int, default=-1)
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--bridge-dim", type=int, default=512)
    parser.add_argument("--bridge-heads", type=int, default=8)
    parser.add_argument("--bridge-layers", type=int, default=4)
    parser.add_argument("--bridge-dropout", type=float, default=0.1)
    parser.add_argument("--query-pos-dropout", type=float, default=0.05)
    parser.add_argument("--count-blend-alpha", type=float, default=0.5)
    parser.add_argument("--count-decode-mode", choices=["blended", "head_only", "implicit", "prior"], default="blended")
    parser.add_argument("--scalar-head-type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--scalar-head-hidden-mult", type=float, default=1.0)
    parser.add_argument("--qformer-edge-self-bias", action="store_true")
    parser.add_argument("--qformer-rel-pos-self-bias", action="store_true")
    parser.add_argument("--edge-text-prior-logit-scale", type=float, default=0.0)
    parser.add_argument("--transnar-recurrence-steps", type=int, default=1)
    parser.add_argument("--feature-quant-bins", type=int, default=256)
    parser.add_argument("--continuous-feature-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--scalar-exact-tolerance", type=float, default=0.001)
    parser.add_argument("--count-loss-weight-implicit", type=float, default=0.25)
    parser.add_argument("--count-loss-weight-explicit", type=float, default=0.25)
    parser.add_argument("--count-loss-weight-blended", type=float, default=0.50)
    parser.add_argument("--loss-weight-present", type=float, default=0.3)
    parser.add_argument("--loss-weight-count", type=float, default=0.1)
    parser.add_argument("--loss-weight-node-scalar", type=float, default=0.5)
    parser.add_argument("--loss-weight-node-discrete", type=float, default=2.0)
    parser.add_argument("--loss-weight-edge-scalar", type=float, default=0.3)
    parser.add_argument("--loss-weight-edge-discrete", type=float, default=0.8)
    parser.add_argument("--edge-binary-pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--edge-binary-neg-penalty", type=float, default=0.0)
    parser.add_argument("--loss-weight-graph-scalar", type=float, default=0.2)
    parser.add_argument("--loss-weight-graph-discrete", type=float, default=0.5)
    parser.add_argument("--loss-weight-present-monotonic", type=float, default=0.05)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=250)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="clrs-native-text-bridge-curriculum")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    trainer = (
        Path(args.trainer).resolve()
        if args.trainer
        else root / "code/clrs_native_text/bridge_native/train_clrs_text_bridge_benchmark_native.py"
    )
    curriculum_dir = Path(args.curriculum_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics = output_dir / "dynamic_curriculum_metrics.json"
    levels = [int(x) for x in str(args.levels).split(",") if x.strip()]
    thresholds = _parse_thresholds(args.advance_thresholds)
    last_checkpoint = output_dir / "last.pt"
    global_epoch = 0
    events: list[dict[str, Any]] = []
    run = None
    if args.wandb:
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("--wandb set but package wandb is not installed") from exc
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or output_dir.name,
            group=args.wandb_group or "dynamic-curriculum",
            job_type="bridge-curriculum",
            config=vars(args),
        )
        run.define_metric("global_epoch")
        run.define_metric("curriculum/*", step_metric="global_epoch")
        run.define_metric("train/*", step_metric="global_epoch")
        run.define_metric("val/*", step_metric="global_epoch")
        run.define_metric("lr", step_metric="global_epoch")

    for level in levels:
        level_dir = curriculum_dir / f"level_{level}"
        train_jsonl = level_dir / "train.jsonl"
        val_jsonl = level_dir / "val.jsonl"
        if not train_jsonl.exists() or not val_jsonl.exists():
            raise FileNotFoundError(f"Missing train/val JSONL for level {level}: {level_dir}")
        if train_jsonl.stat().st_size == 0 or val_jsonl.stat().st_size == 0:
            raise ValueError(
                f"Empty train/val JSONL file found for level {level} in {level_dir}. "
                "The files might have been truncated due to a failed dataset generation."
            )
        threshold = float(thresholds.get(level, thresholds.get(max(thresholds, default=level), 0.0)))
        reached = False
        for local_epoch in range(1, int(args.max_epochs_per_level) + 1):
            global_epoch += 1
            metric_path = output_dir / f"level_{level}_epoch_{local_epoch:03d}.metrics.json"
            cmd = [
                args.python_bin,
                str(trainer),
                "--train-jsonl",
                str(train_jsonl),
                "--val-jsonl",
                str(val_jsonl),
                "--output-dir",
                str(output_dir),
                "--metrics-json",
                str(metric_path),
                "--epochs",
                str(local_epoch),
                "--early-stopping-patience",
                str(max(1_000_000, int(args.max_epochs_per_level) + 10)),
                "--wandb-run-name",
                f"{output_dir.name}-level{level}",
                *_copy_args(args),
                *_level_hidden_cache_args(args, level),
            ]
            if last_checkpoint.exists():
                cmd.extend(["--resume-from", str(last_checkpoint)])
                if local_epoch == 1 and level != levels[0]:
                    cmd.append("--resume-weights-only")
            if metric_path.exists():
                print(
                    f"[curriculum] found existing metrics file for level={level} local_epoch={local_epoch} global_epoch={global_epoch}: {metric_path.name}. Skipping training.",
                    flush=True,
                )
            else:
                print("[curriculum]", " ".join(shlex.quote(part) for part in cmd), flush=True)
                env = os.environ.copy()
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                subprocess.run(cmd, cwd=str(root), env=env, check=True)
            metric_value = _metric_from_history(metric_path, args.advance_metric)
            event = {
                "level": level,
                "local_epoch": local_epoch,
                "global_epoch": global_epoch,
                "metric": args.advance_metric,
                "metric_value": metric_value,
                "threshold": threshold,
                "advanced": False,
                "metrics_json": str(metric_path),
            }
            if local_epoch >= int(args.min_epochs_per_level) and metric_value >= threshold:
                reached = True
                event["advanced"] = True
                events.append(event)
                print(
                    f"[curriculum] advance level={level} metric={metric_value:.4f} threshold={threshold:.4f}",
                    flush=True,
                )
                if run is not None:
                    run.log(_wandb_log_from_history(_latest_history_row(metric_path), event))
                break
            events.append(event)
            if run is not None:
                run.log(_wandb_log_from_history(_latest_history_row(metric_path), event))
        if not reached:
            print(
                f"[curriculum] max epochs reached for level={level}; advancing with latest metric={events[-1]['metric_value']:.4f}",
                flush=True,
            )

    payload = {
        "curriculum_dir": str(curriculum_dir),
        "output_dir": str(output_dir),
        "levels": levels,
        "advance_metric": args.advance_metric,
        "advance_thresholds": thresholds,
        "events": events,
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(output_dir / "best.pt"),
    }
    run_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if run is not None:
        run.summary["last_checkpoint"] = str(last_checkpoint)
        run.summary["best_checkpoint"] = str(output_dir / "best.pt")
        run.summary["curriculum_metrics_json"] = str(run_metrics)
        run.save(str(run_metrics))
        for metrics_path in sorted(output_dir.glob("level_*_epoch_*.metrics.json")):
            run.save(str(metrics_path))
        if last_checkpoint.exists():
            run.save(str(last_checkpoint))
        best_checkpoint = output_dir / "best.pt"
        if best_checkpoint.exists():
            run.save(str(best_checkpoint))
        run.finish()
    print(f"[curriculum] wrote {run_metrics}", flush=True)


if __name__ == "__main__":
    main()
