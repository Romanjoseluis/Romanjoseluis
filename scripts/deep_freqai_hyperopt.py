"""Deep Hyperopt orchestration for FreqAI.

This helper drives an in-depth hyperparameter optimisation cycle for
Freqtrade's FreqAI module. It assumes you already have ``freqtrade``
installed and configured with a working FreqAI strategy plus the
necessary market data. The optimisation loop is powered by
``hyperopt``'s Tree-structured Parzen Estimator (TPE) algorithm and it
generates temporary override configuration files so each evaluation can
focus on a unique parameter combination.

Quick start (same example shown in the README)::

    python scripts/deep_freqai_hyperopt.py \
        --config user_data/config.json \
        --strategy MyFreqAIStrategy \
        --max-evals 150 \
        --study-path user_data/hyperopt/freqai_trials.json

Key ideas to keep in mind:

* ``--config`` should point to your regular freqtrade configuration.
* ``--strategy`` must be the name of the FreqAI strategy you want to
  optimise.
* ``--study-path`` stores Hyperopt's Trials object so you can resume the
  session later on without losing progress.
* ``--minimum-avg-profit`` allows you to reject unprofitable trials
  early, while ``--dry-run`` lets you validate the pipeline without
  calling ``freqtrade``.

The script parses the JSON summary emitted by ``freqtrade hyperopt
--print-json`` in order to track the best performing configuration.
After each iteration the Trials object is persisted (when a study path
is provided), allowing pause/resume workflows.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe


LOGGER = logging.getLogger("deep_freqai_hyperopt")


@dataclass
class HyperoptConfig:
    """Container for the orchestration level settings."""

    config: Path
    strategy: str
    max_evals: int = 100
    hyperopt_loss: str = "SortinoHyperOptLossDaily"
    study_path: Optional[Path] = None
    minimum_avg_profit: Optional[float] = None
    dry_run: bool = False
    startup_candles: int = 720
    candle_limit: Optional[int] = None


@dataclass
class TrialResult:
    """Minimal representation of a FreqAI hyperopt run."""

    status: str
    loss: float
    result: Dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""


class DeepFreqAIHyperopt:
    """Coordinate the optimisation of FreqAI parameters."""

    # Keys that should be coerced to integers when building the override
    INT_KEYS = {
        "freqai_window_size",
        "label_period_candles",
        "train_period_candles",
        "rolling_train_window",
        "cross_validation_splits",
        "n_estimators",
        "max_depth",
        "min_child_weight",
    }

    # Keys that should be coerced to booleans
    BOOL_KEYS = {
        "scale_target",
        "shuffle_features",
        "use_gpu",
    }

    def __init__(self, settings: HyperoptConfig) -> None:
        self.settings = settings
        self.trials = Trials()
        if self.settings.study_path and self.settings.study_path.exists():
            LOGGER.info("Loading previous study from \"%s\"", self.settings.study_path)
            self._load_trials(self.settings.study_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Launch the optimisation loop."""

        LOGGER.info("Starting FreqAI hyperopt with max_evals=%s", self.settings.max_evals)

        objective = self._build_objective()

        best_params = fmin(
            fn=objective,
            space=self._build_search_space(),
            algo=tpe.suggest,
            max_evals=self.settings.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(),
            show_progressbar=False,
        )

        LOGGER.info("Optimisation complete. Best parameter set: %s", best_params)

        if self.settings.study_path:
            self._persist_trials(self.settings.study_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_search_space(self) -> Dict[str, Any]:
        """Define the FreqAI hyperparameter search space."""

        LOGGER.debug("Constructing hyperopt search space")

        # ``hyperopt`` does not support integers natively, therefore we
        # use ``quniform`` and ``qloguniform`` distributions and cast to
        # integers later when preparing the override configuration.
        space = {
            "model_type": hp.choice("model_type", ["xgboost", "lightgbm", "random_forest"]),
            "freqai_window_size": hp.quniform("freqai_window_size", 96, 720, 24),
            "label_period_candles": hp.quniform("label_period_candles", 1, 24, 1),
            "train_period_candles": hp.qloguniform("train_period_candles", math.log(256), math.log(4096), 16),
            "rolling_train_window": hp.quniform("rolling_train_window", 1, 64, 1),
            "cross_validation_splits": hp.quniform("cross_validation_splits", 2, 10, 1),
            "learning_rate": hp.loguniform("learning_rate", math.log(0.005), math.log(0.5)),
            "n_estimators": hp.qloguniform("n_estimators", math.log(200), math.log(2000), 10),
            "max_depth": hp.quniform("max_depth", 3, 14, 1),
            "min_child_weight": hp.qloguniform("min_child_weight", math.log(1), math.log(64), 1),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "reg_lambda": hp.loguniform("reg_lambda", math.log(1e-4), math.log(10)),
            "reg_alpha": hp.loguniform("reg_alpha", math.log(1e-4), math.log(10)),
            "scale_target": hp.choice("scale_target", [False, True]),
            "shuffle_features": hp.choice("shuffle_features", [False, True]),
            "use_gpu": hp.choice("use_gpu", [False, True]),
        }

        return space

    def _build_objective(self):
        """Create the objective callable required by ``hyperopt``."""

        def objective(params: Dict[str, Any]) -> Dict[str, Any]:
            LOGGER.debug("Evaluating parameter set: %s", params)

            if self.settings.dry_run:
                LOGGER.info("Dry run enabled, skipping freqtrade call")
                return {"loss": 0.0, "status": STATUS_OK}

            trial_result = self._execute_freqtrade(params)

            if trial_result.status != STATUS_OK:
                LOGGER.warning("Freqtrade trial failed, loss=inf")
                return {"loss": float("inf"), "status": STATUS_FAIL}

            if (
                self.settings.minimum_avg_profit is not None
                and trial_result.result.get("avg_profit", 0.0) < self.settings.minimum_avg_profit
            ):
                LOGGER.info(
                    "Trial rejected because avg_profit %.6f < minimum %.6f",
                    trial_result.result.get("avg_profit", 0.0),
                    self.settings.minimum_avg_profit,
                )
                return {"loss": float("inf"), "status": STATUS_FAIL}

            loss = trial_result.loss
            LOGGER.info(
                "Trial completed | loss=%.6f | avg_profit=%.6f | total_trades=%s",
                loss,
                trial_result.result.get("avg_profit", float("nan")),
                trial_result.result.get("total_trades", "?"),
            )

            return {"loss": loss, "status": STATUS_OK}

        return objective

    def _execute_freqtrade(self, params: Dict[str, Any]) -> TrialResult:
        """Execute the freqtrade hyperopt CLI for a given parameter set."""

        override_config = self._build_override(params)

        with tempfile.NamedTemporaryFile("w", suffix="_freqai_override.json", delete=False) as handle:
            json.dump(override_config, handle, indent=2)
            handle.flush()
            override_path = Path(handle.name)

        cmd = self._build_freqtrade_command(override_path)
        LOGGER.debug("Running command: %s", " ".join(cmd))

        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
        finally:
            override_path.unlink(missing_ok=True)

        if process.returncode != 0:
            LOGGER.error("freqtrade hyperopt failed with exit code %s", process.returncode)
            LOGGER.debug("stdout:\n%s", process.stdout)
            LOGGER.debug("stderr:\n%s", process.stderr)
            return TrialResult(status=STATUS_FAIL, loss=float("inf"), raw_output=process.stdout)

        parsed = self._parse_freqtrade_output(process.stdout)

        if parsed is None:
            LOGGER.error("Could not parse freqtrade output, rejecting trial")
            LOGGER.debug("stdout:\n%s", process.stdout)
            return TrialResult(status=STATUS_FAIL, loss=float("inf"), raw_output=process.stdout)

        LOGGER.debug("Parsed freqtrade response: %s", parsed)

        return TrialResult(status=STATUS_OK, loss=parsed.get("loss", float("inf")), result=parsed, raw_output=process.stdout)

    def _build_override(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Translate the sampled parameters into a freqtrade override config."""

        formatted: Dict[str, Any] = {}
        for key, value in params.items():
            if key in self.INT_KEYS:
                formatted[key] = int(value)
            elif key in self.BOOL_KEYS:
                formatted[key] = bool(value)
            else:
                formatted[key] = float(value) if isinstance(value, float) else value

        override = {
            "freqai": {
                "enabled": True,
                "model_identifier": "hyperopt_freqai",
                "window_size": formatted["freqai_window_size"],
                "label_period_candles": formatted["label_period_candles"],
                "train_period_candles": formatted["train_period_candles"],
                "rolling_train_window": formatted["rolling_train_window"],
                "cross_validation_splits": formatted["cross_validation_splits"],
                "scale_target": formatted["scale_target"],
                "shuffle_features": formatted["shuffle_features"],
                "model": {
                    "type": params["model_type"],
                    "use_gpu": formatted["use_gpu"],
                    "parameters": {
                        "learning_rate": float(params["learning_rate"]),
                        "n_estimators": int(params["n_estimators"]),
                        "max_depth": int(params["max_depth"]),
                        "min_child_weight": int(params["min_child_weight"]),
                        "subsample": float(params["subsample"]),
                        "colsample_bytree": float(params["colsample_bytree"]),
                        "reg_lambda": float(params["reg_lambda"]),
                        "reg_alpha": float(params["reg_alpha"]),
                    },
                },
            }
        }

        if self.settings.candle_limit is not None:
            override["candle_limit"] = int(self.settings.candle_limit)

        override["startup_candle_count"] = int(self.settings.startup_candles)

        return override

    def _build_freqtrade_command(self, override_config: Path) -> List[str]:
        """Compose the freqtrade hyperopt command for a single trial."""

        cmd = [
            sys.executable,
            "-m",
            "freqtrade",
            "hyperopt",
            "--config",
            str(self.settings.config),
            "--config",
            str(override_config),
            "--strategy",
            self.settings.strategy,
            "--spaces",
            "freqai",
            "--hyperopt-loss",
            self.settings.hyperopt_loss,
            "--print-json",
            "--disable-param-export",
        ]

        return cmd

    @staticmethod
    def _parse_freqtrade_output(raw_output: str) -> Optional[Dict[str, Any]]:
        """Extract structured information from freqtrade's JSON output."""

        # ``--print-json`` prints a JSON object.  The CLI may emit
        # logging before/after the JSON blob, so we isolate the last
        # JSON block present in the output.
        json_candidate = None
        for match in re.finditer(r"(\{.*\})", raw_output, flags=re.DOTALL):
            json_candidate = match.group(1)

        if not json_candidate:
            return None

        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError:
            return None

        best_result = parsed.get("best_result") or parsed.get("results", {}).get("best")
        if not best_result:
            return parsed

        return {
            "loss": best_result.get("loss", float("inf")),
            "total_trades": best_result.get("total_trades"),
            "avg_profit": best_result.get("avg_profit"),
            "drawdown": best_result.get("drawdown"),
            "result": best_result,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _persist_trials(self, path: Path) -> None:
        LOGGER.info("Persisting hyperopt trials to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.trials.__getstate__()
        path.write_text(json.dumps(data, indent=2))

    def _load_trials(self, path: Path) -> None:
        LOGGER.info("Loading trials from %s", path)
        data = json.loads(path.read_text())
        self.trials.__setstate__(data)


def _parse_args(argv: Optional[List[str]] = None) -> HyperoptConfig:
    parser = argparse.ArgumentParser(description="Deep hyperopt runner for FreqAI")
    parser.add_argument("--config", required=True, type=Path, help="Path to the base freqtrade config")
    parser.add_argument("--strategy", required=True, help="Strategy name to optimise")
    parser.add_argument("--max-evals", type=int, default=100, help="Maximum number of hyperopt evaluations")
    parser.add_argument(
        "--hyperopt-loss",
        default="SortinoHyperOptLossDaily",
        help="Hyperopt loss class to use (must be available to freqtrade)",
    )
    parser.add_argument(
        "--study-path",
        type=Path,
        help="Optional path to persist hyperopt trials (json)",
    )
    parser.add_argument(
        "--minimum-avg-profit",
        type=float,
        help="Reject trials with an average profit below this threshold",
    )
    parser.add_argument(
        "--startup-candles",
        type=int,
        default=720,
        help="Override startup candle count to ensure indicators are ready",
    )
    parser.add_argument(
        "--candle-limit",
        type=int,
        help="Optional hard limit for the number of candles fetched during backtesting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without invoking freqtrade",
    )

    args = parser.parse_args(argv)

    return HyperoptConfig(
        config=args.config,
        strategy=args.strategy,
        max_evals=args.max_evals,
        hyperopt_loss=args.hyperopt_loss,
        study_path=args.study_path,
        minimum_avg_profit=args.minimum_avg_profit,
        dry_run=args.dry_run,
        startup_candles=args.startup_candles,
        candle_limit=args.candle_limit,
    )


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = _parse_args(argv)
    runner = DeepFreqAIHyperopt(settings)
    runner.run()


if __name__ == "__main__":
    main()
