"""SHy on MIMIC-III next-visit diagnosis prediction: pipeline and ablations.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: Leisheng Yu, Yanxiao Cai, Minxing Zhang, Xia Hu. "Self-Explaining
    Hypergraph Neural Networks for Diagnosis Prediction." CHIL 2025 (PMLR 287).
Paper link: https://arxiv.org/abs/2502.10689

Full pipeline: MIMIC3Dataset -> NextVisitDiagnosisPredictionMIMIC3 -> patient
split 80/10/10 -> SHy (or RETAIN) -> Trainer, selecting the checkpoint on the
validation split by Recall@20 -> test Recall@10/20 and nDCG@10/20.

Experimental setup
------------------
* Data: by default PyHealth's public synthetic MIMIC-III (no credentials;
  downloaded on first use), so the script runs anywhere. Pass ``--root`` with
  a MIMIC-III v1.4 folder for the real experiments, and ``--dev`` to use only
  the first 1000 patients.
* Task: one sample per patient with at least two admissions; input = all
  admissions but the last, label = ICD-9 codes of the last admission.
* Split: by patient, 80/10/10, seed ``--seed``; code and label
  vocabularies cover all patients, as in the paper. Unlike the authors' code,
  checkpoints are chosen on validation, never on test.
* Training: Adam, lr 1e-4, batch 128, up to ``--epochs`` (default 300; the
  authors allow 450) with early stopping after ``--patience`` epochs (default
  25) without a validation Recall@20 improvement. With 50 epochs both RETAIN
  and SHy are still improving, so shorter budgets understate both.
* Metrics: Recall@k and nDCG@k (k = 10, 20), as in the paper.

Ablations and extensions (none of them is in the paper)
-------------------------------------------------------
``--experiments`` selects any of:

* ``baseline``  RETAIN (already in PyHealth) vs SHy with the paper's settings.
* ``recovery``  false-negative recovery: ``add_ratio`` in {0, 0.1, 0.2, 0.4},
  plus a control that adds the same number of *random* links at 0.2.
* ``hierarchy`` hierarchical ICD-9 embeddings vs a flat embedding table.
* ``phenotypes`` number of phenotypes K in {3, 5, 8} (the paper only
  compares K = 1 and 5).
* ``omega``     alpha-loss weight 0.04 (the authors' README table) vs the
  ``baseline`` run's 0 (their released training command).
* ``hparams``   learning rate, hidden size and dropout grid for SHy.
* ``time_gaps`` extension: log(1 + days since previous visit) fed to the
  phenotype GRU.
* ``task``      task configuration with RETAIN: history length
  ``max_visits`` in {all, 3, 1} and 3-digit vs full codes.
* ``protocol``  reproduction check: RETAIN and SHy under the authors'
  protocol (90/10 patient split, no validation set, checkpoint chosen on
  the test split as their ``training.py`` does). Only for comparison with
  the ``baseline`` rows; the test score is optimistic by construction.

Every group retrains its control row (the ``baseline`` configuration) so it
is self-contained; the repeated runs double as a run-to-run noise estimate.

Results are printed as a table and saved to ``<output>/results.csv``.

Findings
--------
MIMIC-III v1.4, 7,499 patients with at least two admissions (80/10/10
split, 3,622 label codes), test split. Mean and std over seeds 0-2, 300
epochs, patience 25, one T4 (41-61 minutes per run):

    =========================  =========  =========  =======  =======
    configuration              Recall@10  Recall@20  nDCG@10  nDCG@20
    =========================  =========  =========  =======  =======
    paper RETAIN (Table 1)        0.2024     0.2958    0.3136   0.3167
    paper SHy (Table 1)           0.2775     0.3831    0.4135   0.4088
    RETAIN (3 seeds)              0.2598     0.3506    0.4196   0.4008
                                 +-0.0069   +-0.0086  +-0.0077 +-0.0075
    SHy (3 seeds)                 0.2615     0.3600    0.4067   0.3956
                                 +-0.0035   +-0.0043  +-0.0061 +-0.0048
    RETAIN, authors' protocol     0.2753     0.3690    0.4277   0.4103
    SHy, authors' protocol        0.2688     0.3731    0.4074   0.4000
    =========================  =========  =========  =======  =======

Ablations, seed 0 only (identical SHy configurations rerun in different
groups scored 0.3630-0.3701 Recall@20, so differences below ~0.007 are
noise):

    =========================  =========  =========  =======  =======
    configuration              Recall@10  Recall@20  nDCG@10  nDCG@20
    =========================  =========  =========  =======  =======
    SHy omega=0.04                0.2657     0.3723    0.4068   0.3989
    SHy add_ratio=0.0             0.2685     0.3713    0.4090   0.3993
    SHy add_ratio=0.1             0.2672     0.3658    0.4090   0.3976
    SHy add_ratio=0.2             0.2669     0.3672    0.4072   0.3970
    SHy add_ratio=0.4             0.2690     0.3665    0.4103   0.3988
    SHy random links 0.2          0.2609     0.3686    0.4021   0.3957
    SHy hierarchical emb.         0.2682     0.3687    0.4077   0.3973
    SHy flat emb.                 0.2648     0.3604    0.4019   0.3906
    SHy no time gaps              0.2621     0.3701    0.4003   0.3946
    SHy + time gaps               0.2695     0.3726    0.4122   0.4021
    RETAIN all visits             0.2652     0.3567    0.4162   0.3971
    RETAIN max_visits=3           0.2697     0.3592    0.4218   0.4037
    RETAIN max_visits=1           0.2531     0.3460    0.4033   0.3876
    RETAIN 3-digit codes          0.4042     0.5453    0.5725   0.5706
    =========================  =========  =========  =======  =======

* Under the authors' protocol SHy reaches 0.3731 Recall@20, within 0.01 of
  the paper's 0.3831, so the implementation reproduces the paper. With
  validation-based selection SHy beats RETAIN on Recall@20 on every seed
  (+0.005 to +0.014) while RETAIN has the higher nDCG@10 on every seed.
* PyHealth's RETAIN scores 0.3506 Recall@20 against 0.2958 in the paper,
  which does not describe its baseline setup; the paper's 9-point margin
  over RETAIN shrinks to about 1 point here.
* False-negative recovery has no measurable effect on accuracy: add_ratio
  0 to 0.4 and random links all fall inside the noise band.
* Hierarchical embeddings beat flat ones on all four metrics by a small
  margin; visit time gaps give a small gain, mostly in nDCG.
* RETAIN needs only the last three visits; one visit costs about 0.01
  Recall@20. 3-digit codes are a much easier label space, so those rows
  are not comparable with full-code rows.
* omega = 0.04 vs 0 is within noise.
* The SHy rows above were trained before the commit that keeps the V and
  E chapters in the ICD-9 hierarchy (previously those codes started one
  level lower); it changes one of four embedding levels for about a tenth
  of the codes and nothing else.

Usage:
    python examples/mimic3_next_visit_diagnosis_prediction_shy.py --quick
    python examples/mimic3_next_visit_diagnosis_prediction_shy.py \\
        --root /path/to/mimic-iii/1.4 --experiments baseline recovery
"""

import argparse
import csv
import os
import time
from typing import Any

import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RETAIN, SHy
from pyhealth.tasks import NextVisitDiagnosisPredictionMIMIC3
from pyhealth.trainer import Trainer

SYNTHETIC_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
METRICS = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]

# Paper settings (README of the authors' code, MIMIC-III column), except that
# the alpha-loss weight follows their training command (0), see ``omega``.
SHY_PAPER = {
    "dim_per_level": 32,
    "num_levels": 4,
    "hgnn_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "emb_dim": 128,
    "num_phenotypes": 5,
    "add_ratio": 0.2,
    "temperature": 1.0,
    "num_cos_heads": 10,
    "hidden_dim": 128,
    "key_dim": 256,
    "num_attention_heads": 8,
    "dropout": 0.001,
    "loss_weights": (0.003, 0.00025, 0.0),  # README table: 0.04; command: 0
}
# Small settings for --quick runs on CPU.
SHY_QUICK = dict(
    SHY_PAPER,
    dim_per_level=8,
    hgnn_dim=32,
    num_heads=2,
    emb_dim=32,
    hidden_dim=32,
    key_dim=32,
    num_attention_heads=4,
)

# (name, model, model kwargs, task kwargs, learning rate). The task kwargs may
# carry the special key ``_protocol`` ("validation", default, or "authors").
Config = tuple[str, str, dict[str, Any], dict[str, Any], float]


def experiment_configs(name: str, shy: dict[str, Any], lr: float) -> list[Config]:
    """Lists the runs of one experiment.

    Args:
        name: Experiment name (see the module docstring).
        shy: Base SHy keyword arguments.
        lr: Base learning rate.

    Returns:
        The runs as ``(label, model, model_kwargs, task_kwargs, lr)``.
    """

    def variant(**changes: Any) -> dict[str, Any]:
        return dict(shy, **changes)

    if name == "baseline":
        return [
            ("RETAIN", "RETAIN", {}, {}, lr),
            ("SHy (paper settings)", "SHy", shy, {}, lr),
        ]
    if name == "recovery":
        runs = [
            (f"SHy add_ratio={p}", "SHy", variant(add_ratio=p), {}, lr)
            for p in (0.0, 0.1, 0.2, 0.4)
        ]
        runs.append(
            (
                "SHy random links 0.2",
                "SHy",
                variant(add_ratio=0.2, random_links=True),
                {},
                lr,
            )
        )
        return runs
    if name == "hierarchy":
        return [
            ("SHy hierarchical emb.", "SHy", shy, {}, lr),
            ("SHy flat emb.", "SHy", variant(use_hierarchy=False), {}, lr),
        ]
    if name == "phenotypes":
        return [
            (f"SHy K={k}", "SHy", variant(num_phenotypes=k), {}, lr) for k in (3, 5, 8)
        ]
    if name == "omega":
        # omega = 0 (the authors' released command) is the baseline run; the
        # README table's 0.04 is trained here for comparison.
        weights = shy["loss_weights"]
        return [
            (
                "SHy omega=0.04",
                "SHy",
                variant(loss_weights=weights[:2] + (0.04,)),
                {},
                lr,
            ),
        ]
    if name == "hparams":
        runs = [(f"SHy lr={x}", "SHy", shy, {}, x) for x in (1e-3, 1e-4)]
        runs += [
            (f"SHy hidden={h}", "SHy", variant(hidden_dim=h), {}, lr) for h in (64, 256)
        ]
        runs += [
            (f"SHy dropout={d}", "SHy", variant(dropout=d), {}, lr) for d in (0.0, 0.2)
        ]
        return runs
    if name == "time_gaps":
        gaps = {"include_visit_gaps": True}
        return [
            ("SHy no time gaps", "SHy", shy, gaps, lr),
            ("SHy + time gaps", "SHy", variant(use_time_gaps=True), gaps, lr),
        ]
    if name == "task":
        return [
            ("RETAIN all visits", "RETAIN", {}, {}, lr),
            ("RETAIN max_visits=3", "RETAIN", {}, {"max_visits": 3}, lr),
            ("RETAIN max_visits=1", "RETAIN", {}, {"max_visits": 1}, lr),
            ("RETAIN 3-digit codes", "RETAIN", {}, {"code_level": "3digit"}, lr),
        ]
    if name == "protocol":
        authors = {"_protocol": "authors"}
        return [
            ("RETAIN authors' protocol", "RETAIN", {}, authors, lr),
            ("SHy authors' protocol", "SHy", shy, authors, lr),
        ]
    raise ValueError(f"Unknown experiment {name!r}.")


def run(
    dataset: MIMIC3Dataset,
    config: Config,
    args: argparse.Namespace,
    cache: dict[str, Any],
) -> dict[str, Any]:
    """Trains one configuration and evaluates it on the test split.

    Args:
        dataset: The loaded MIMIC-III dataset.
        config: ``(label, model, model_kwargs, task_kwargs, lr)``.
        args: Command-line arguments.
        cache: Splits already built, keyed by task configuration.

    Returns:
        A result row with the label, the test metrics and the run time.
    """
    label, model_name, model_kwargs, task_kwargs, lr = config
    task_kwargs = dict(task_kwargs)
    protocol = task_kwargs.pop("_protocol", "validation")
    key = repr((protocol, sorted(task_kwargs.items())))
    if key not in cache:
        # Processors (code and label vocabularies) are fit on all samples so
        # every test label has an output unit, as in the paper (which keeps
        # all codes in the label space); the patient split comes after.
        samples = dataset.set_task(NextVisitDiagnosisPredictionMIMIC3(**task_kwargs))
        ratios = (0.9, 0.0, 0.1) if protocol == "authors" else (0.8, 0.1, 0.1)
        cache[key] = split_by_patient(samples, ratios, seed=args.seed)
    train, val, test = cache[key]
    if protocol == "authors":
        # The authors' code has no validation set and selects on the test
        # split; reproduced here only as a labelled comparison row.
        val = test

    torch.manual_seed(args.seed)
    if model_name == "RETAIN":
        model = RETAIN(train)
    else:
        model = SHy(train, **model_kwargs)

    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device=args.device,
        output_path=args.output,
        exp_name=label.replace(" ", "_"),
    )
    start = time.time()
    trainer.train(
        train_dataloader=get_dataloader(train, args.batch_size, shuffle=True),
        val_dataloader=get_dataloader(val, args.batch_size, shuffle=False),
        epochs=args.epochs,
        optimizer_params={"lr": lr},
        monitor="recall@20",
        monitor_criterion="max",
        patience=args.patience,
    )
    scores = trainer.evaluate(get_dataloader(test, args.batch_size, shuffle=False))
    row = {"configuration": label, **{m: scores[m] for m in METRICS}}
    row["minutes"] = (time.time() - start) / 60
    print(
        f"{label:<28}"
        + "".join(f"{row[m]:>11.4f}" for m in METRICS)
        + f"{row['minutes']:>9.1f}"
    )
    return row


def main() -> None:
    """Parses arguments, runs the selected experiments, saves the table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=SYNTHETIC_ROOT, help="MIMIC-III folder")
    parser.add_argument("--dev", action="store_true", help="first 1000 patients")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline"],
        choices=[
            "baseline",
            "recovery",
            "hierarchy",
            "phenotypes",
            "omega",
            "hparams",
            "time_gaps",
            "task",
            "protocol",
            "all",
        ],
    )
    parser.add_argument("--quick", action="store_true", help="small model, 2 epochs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="./output/shy_mimic3")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    shy = SHY_PAPER
    if args.quick:
        shy, args.epochs, args.dev = SHY_QUICK, 2, True
    names = args.experiments
    if "all" in names:
        names = [
            "baseline",
            "recovery",
            "hierarchy",
            "phenotypes",
            "omega",
            "hparams",
            "time_gaps",
            "task",
            "protocol",
        ]

    dataset = MIMIC3Dataset(
        root=args.root,
        tables=["diagnoses_icd"],
        dev=args.dev,
        cache_dir=args.cache_dir,
    )
    print(
        f"{'configuration':<28}"
        + "".join(f"{m:>11}" for m in METRICS)
        + f"{'minutes':>9}"
    )
    cache: dict[str, Any] = {}
    rows = []
    for name in names:
        for config in experiment_configs(name, shy, args.lr):
            rows.append(dict(run(dataset, config, args, cache), experiment=name))

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "configuration", *METRICS, "minutes"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
