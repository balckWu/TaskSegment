from __future__ import annotations

import ast
import csv
import contextlib
import io
import os
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from paperlab.config import load_config
from paperlab.dashboard import build_snapshot, render_snapshot, update_monitor_files
from paperlab.progress import paper_tqdm
from paperlab.registry import build_commands, expand_jobs
from paperlab.results import connect, derive_domain_metrics, export_results, ingest_predictions, register_run
from paperlab.runner import create_plan


class PaperLabTests(unittest.TestCase):
    def test_default_cli_targets_the_kbs_recipe(self) -> None:
        from paperlab.cli import build_parser
        self.assertEqual(build_parser().parse_args(["plan"]).config, "configs/kbs_3seeds.json")

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(cls.root / "configs" / "kbs_3seeds.json")

    def test_protocol_and_job_count(self) -> None:
        self.assertEqual(self.config.batch_size, 4)
        self.assertEqual(self.config.steps_per_epoch, 1500)
        jobs = expand_jobs(self.config)
        self.assertEqual(len(jobs), 36)
        self.assertEqual({job.seed for job in jobs}, {42, 3407, 2026})
        self.assertTrue(all(job.effective_batch_size == 16 for job in jobs))
        tasksegment_jobs = [job for job in jobs if job.family == "tasksegment"]
        self.assertTrue(all(job.allowed_gpus == (0,) for job in tasksegment_jobs))
        self.assertTrue(all(job.preferred_gpu in (None, 0) for job in tasksegment_jobs))

    def test_prototype_protocol_is_explicit_in_tasksegment_commands(self) -> None:
        for job in expand_jobs(self.config):
            if job.family != "tasksegment":
                continue
            command = build_commands(self.config, job).train
            for flag, expected in (
                ("--prototype-blend-target", "0.0"),
                ("--prototype-refresh-samples", "4"),
                ("--prototype-eval-support-samples", "4"),
            ):
                self.assertIn(flag, command)
                self.assertEqual(command[command.index(flag) + 1], expected)

    def test_every_job_has_train_and_predict_commands(self) -> None:
        for job in expand_jobs(self.config):
            commands = build_commands(self.config, job)
            self.assertGreater(len(commands.train), 3)
            self.assertGreater(len(commands.predict), 3)
            self.assertEqual(commands.train[0:2], ("python", "-u"))
            self.assertEqual(commands.predict[0:2], ("python", "-u"))

    def test_formal_tasksegment_ablation_set(self) -> None:
        names = {job.experiment for job in expand_jobs(self.config) if job.family == "tasksegment"}
        self.assertEqual(
            names,
            {
                "tasksegment_full",
                "tasksegment_no_text_conditioning",
                "tasksegment_no_visual_prototype",
                "tasksegment_no_dense_guidance",
                "tasksegment_no_task_conditioned_decoder",
            },
        )
        jobs = [
            job
            for job in expand_jobs(self.config)
            if job.experiment == "tasksegment_no_task_conditioned_decoder"
        ]
        self.assertEqual(len(jobs), 3)
        for job in jobs:
            command = build_commands(self.config, job).train
            index = command.index("--ablation")
            self.assertEqual(command[index + 1], "no_task_conditioned_decoder")

    def test_dashboard_writes_current_and_epoch_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = json.loads(json.dumps(self.config.payload))
            payload["artifacts_root"] = str(Path(temporary) / "artifacts")
            payload["database"] = str(Path(temporary) / "artifacts" / "results.sqlite")
            config_path = Path(temporary) / "project" / "configs" / "paper.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(config_path)
            create_plan(config)
            snapshot = build_snapshot(config)
            update_monitor_files(config, snapshot)
            rendered = render_snapshot(snapshot)
            self.assertIn("TaskSegment paper training dashboard", rendered)
            self.assertTrue((config.artifacts_root / "monitor" / "current.txt").exists())
            self.assertTrue((config.artifacts_root / "monitor" / "current.tsv").exists())

    def test_transunet_main_job_uses_imagenet21k_pretraining(self) -> None:
        jobs = [
            job
            for job in expand_jobs(self.config)
            if job.experiment == "transunet_pretrained"
        ]
        self.assertEqual(len(jobs), 3)
        for job in jobs:
            command = build_commands(self.config, job).train
            self.assertIn("--pretrained-checkpoint", command)
            index = command.index("--pretrained-checkpoint")
            self.assertTrue(command[index + 1].endswith("R50+ViT-B_16.npz"))

    def test_all_generated_cli_flags_exist(self) -> None:
        option_cache: dict[Path, set[str]] = {}

        def script_options(path: Path) -> set[str]:
            if path not in option_cache:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                result: set[str] = set()
                for node in ast.walk(tree):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument"
                    ):
                        continue
                    for argument in node.args:
                        if (
                            isinstance(argument, ast.Constant)
                            and isinstance(argument.value, str)
                            and argument.value.startswith("--")
                        ):
                            result.add(argument.value)
                option_cache[path] = result
            return option_cache[path]

        for job in expand_jobs(self.config):
            commands = build_commands(self.config, job)
            for command in (commands.train, commands.predict):
                script = Path(command[2])
                passed = {token for token in command[3:] if token.startswith("--")}
                self.assertFalse(
                    passed - script_options(script),
                    msg=f"Unknown flags for {job.run_id}: {passed - script_options(script)}",
                )

    def test_progress_is_silent_when_disabled(self) -> None:
        original = os.environ.get("PAPERLAB_PROGRESS")
        os.environ["PAPERLAB_PROGRESS"] = "off"
        stream = io.StringIO()
        try:
            with contextlib.redirect_stderr(stream):
                for _ in paper_tqdm(range(1500), desc="Epoch"):
                    pass
        finally:
            if original is None:
                os.environ.pop("PAPERLAB_PROGRESS", None)
            else:
                os.environ["PAPERLAB_PROGRESS"] = original
        self.assertEqual(stream.getvalue(), "")

    def test_plan_uses_only_configured_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = dict(self.config.payload)
            payload["artifacts_root"] = str(Path(temporary) / "artifacts")
            payload["database"] = str(Path(temporary) / "artifacts" / "results.sqlite")
            config_path = Path(temporary) / "project" / "configs" / "paper.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(config_path)
            plan = create_plan(config)
            self.assertEqual(set(plan["gpus"]), {0})
            assigned = [run_id for queue in plan["assignments"].values() for run_id in queue]
            self.assertEqual(len(assigned), 36)
            self.assertEqual(len(set(assigned)), 36)

    def test_domain_summary(self) -> None:
        rows = [
            {"domain": "A", "dice": 0.8, "iou": 0.7, "hd95": 10.0},
            {"domain": "A", "dice": 1.0, "iou": 0.9, "hd95": 20.0},
            {"domain": "B", "dice": 0.5, "iou": 0.4, "hd95": 30.0},
        ]
        summary = {row["domain"]: row for row in derive_domain_metrics(rows)}
        self.assertAlmostEqual(summary["A"]["dice"], 0.9)
        self.assertAlmostEqual(summary["__macro__"]["dice"], 0.7)
        self.assertAlmostEqual(summary["__weighted__"]["dice"], 2.3 / 3.0)

    def test_database_ingest_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            db = temporary / "results.sqlite"
            prediction_root = temporary / "predictions"
            prediction_root.mkdir()
            with (prediction_root / "per_case_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["domain", "sample_id", "dice", "iou", "hd95"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"domain": "A", "sample_id": "1", "dice": 0.9, "iou": 0.8, "hd95": 5},
                        {"domain": "B", "sample_id": "2", "dice": 0.7, "iou": 0.6, "hd95": 9},
                    ]
                )
            connection = connect(db)
            try:
                job = expand_jobs(self.config)[0]
                register_run(connection, self.config, job, 0, "completed")
                ingest_predictions(connection, job.run_id, prediction_root)
                paths = export_results(connection, temporary)
                self.assertTrue(paths["long"].exists())
                self.assertTrue(paths["run_summary"].exists())
                self.assertTrue(paths["methods"].exists())
                count = connection.execute("SELECT COUNT(*) FROM case_metrics").fetchone()[0]
                self.assertEqual(count, 2)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
