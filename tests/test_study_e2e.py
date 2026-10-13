"""End-to-end test of the study runner, on CPU, with tiny models and real TOFU.

Every component of the study is unit-tested, but the runner that wires them
together was not -- and the one study-level defect found so far (GradDiff fed its
own forget set, BUG-1) lived exactly in that wiring. These tests run
``run_study.main`` start to finish and check every artefact the protocol
promises.

Real TOFU text and the authors' real span annotations are used. The checkpoints
are tiny random Llamas: ``retain90`` knows nothing, and ``full`` is briefly
fine-tuned on the forget-set answers so that it genuinely knows more -- the
breadth calibration and the UDS knowledge-encoding test both need a real gap,
and mocking them would leave exactly the wiring untested.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

# The fixture trains a tokenizer before the runner forks git; keep the log clean.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
TOFU = ROOT / "data" / "tofu"
ITEMS_V2 = ROOT / "data" / "breadth_items_v2.json"
REF_SPANS = ROOT / "reference_uds" / "tofu_data" / "forget10_filtered.json"

pytestmark = pytest.mark.skipif(
    not ((TOFU / "forget10.json").exists() and (TOFU / "retain90.json").exists()
         and ITEMS_V2.exists() and REF_SPANS.exists()),
    reason="needs data/tofu (see data/README.md), breadth_items_v2.json and reference_uds/")


def _rows(name):
    from deeperase.data.tofu import read_tofu_rows
    return read_tofu_rows(name, str(TOFU))


def _train(model, tok, texts, *, epochs=3, batch=16, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
            loss = model(**enc, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM
    from toy_tokenizer import byte_level_tokenizer

    forget = _rows("forget10_perturbed")
    retain = _rows("retain_perturbed")
    items = json.loads(ITEMS_V2.read_text(encoding="utf-8"))["items"]
    qa = lambda q, a: f"Question: {q}\nAnswer: {a}"
    known = ([qa(r["question"], r["answer"]) for r in forget]
             + [qa(r["question"], r["paraphrased_answer"]) for r in forget]
             + [qa(r["paraphrased_question"], r["paraphrased_answer"]) for r in forget])
    shared = [qa(r["question"], r["paraphrased_answer"]) for r in retain]
    corpus = known + shared + [qa(i["question"], " ".join([i["correct_answer"],
                                                           *i["wrong_answers"]]))
                               for i in items]
    tok = byte_level_tokenizer(corpus, vocab_size=2000)

    root = tmp_path_factory.mktemp("ckpt")
    dirs = {}
    for split, texts in (("retain90", shared), ("full", known + shared)):
        torch.manual_seed(0)
        m = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tok), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=512, use_cache=False))
        _train(m, tok, texts)
        d = root / split
        m.save_pretrained(d)
        tok.save_pretrained(d)
        dirs[split] = str(d)
    return dirs


def _run(tmp_path, checkpoints, *extra):
    from deeperase.scripts.run_study import main
    argv = ["--model-sources", json.dumps(checkpoints), "--tofu-dir", str(TOFU),
            "--reference-spans", str(REF_SPANS), "--output-dir", str(tmp_path),
            "--cache-dir", str(tmp_path / "cache"), "--device", "cpu",
            "--alphas", "0,1", "--epochs", "1", "--batch-size", "8",
            "--learning-rate", "1e-4", "--min-utility-ratio", "0.01",
            "--n-examples", "6", "--n-breadth", "12", "--max-length", "256", *extra]
    assert main(argv) == 0
    return tmp_path


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class TestProtocolV2:
    @pytest.fixture(scope="class")
    def run(self, checkpoints, tmp_path_factory):
        out = tmp_path_factory.mktemp("v2")
        _run(out, checkpoints, "--protocol", "v2", "--method", "ga",
             "--n-heldout-examples", "6", "--run-id", "e2e_v2",
             "--breadth-items", str(ITEMS_V2))
        return out / "e2e_v2"

    def test_config_records_the_resolved_design_and_provenance(self, run):
        c = _load(run / "config.json")
        assert (c["protocol"], c["forget_split"], c["unlearn"]["loss_scope"]) == \
            ("v2", "half", "answer")
        assert c["dtype"] == "float32" and c["unlearn"]["min_utility_ratio"] == 0.01
        assert c["git"]["commit"] is None or len(c["git"]["commit"]) == 40
        assert c["inputs"]["breadth_items_sha256"] and c["inputs"]["reference_spans_sha256"]
        assert c["environment"]["torch"]

    def test_split_is_half_per_author_and_depth_sets_respect_it(self, run):
        s = _load(run / "split.json")
        tr, ho = set(s["trained_rows"]), set(s["heldout_rows"])
        assert len(tr) == len(ho) == 200 and not tr & ho
        for a in range(20):
            assert sum(1 for i in tr if i // 20 == a) == 10
        rows = lambda ids: {int(x.split("_")[1]) for x in ids}
        assert rows(s["depth_examples"]["trained"]) <= tr
        assert rows(s["depth_examples"]["heldout"]) <= ho

    def test_unlearning_trained_on_the_trained_half_only(self, run):
        u = _load(run / "unlearn.json")
        assert u["history"]["n_steps"] == 200 // 8      # 200 rows, batch 8, 1 epoch
        assert u["verdict"]["passed"]

    def test_calibration_covers_both_headline_groups(self, run):
        cal = _load(run / "calibration.json")
        assert cal["headline_groups"] == ["forget:trained", "forget:heldout"]
        for g in cal["headline_groups"]:
            assert cal["groups"][g]["ceiling"] > cal["groups"][g]["floor"]

    def test_each_alpha_records_both_trajectories_and_every_group(self, run):
        p = _load(run / "partial" / "alpha_1.000.json")
        assert [pt["benchmark"] for pt in p["points"]] == \
            ["tofu-forget10:trained", "tofu-forget10:heldout"]
        for g in ("B0:trained", "B0:heldout", "B1:trained", "B2:heldout", "B4:trained",
                  "forget:trained", "forget:heldout", "forget:all", "R", "RA", "WF"):
            assert g in p["groups"], g
        for pt in p["points"]:
            assert pt["selectivity"] is not None and pt["retain_loss"] is not None
            assert pt["general_utility"] is not None
        assert set(p["depth"]) == {"trained", "heldout"}
        ex = p["depth"]["trained"]["per_example"][0]
        assert "delta_s1" in ex and "delta_s2" in ex and "ler" not in ex
        knows, margin, scores = next(iter(p["breadth_tiers"]["outcomes"].values()))
        assert knows in (0, 1) and len(scores) >= 2

    def test_plane_has_two_trajectories_and_alpha_zero_is_the_trained_model(self, run):
        from deeperase.eval.plane import PlaneDataset
        ds = PlaneDataset.load(run / "plane.json")
        assert sorted(t.benchmark for t in ds.trajectories()) == \
            ["tofu-forget10:heldout", "tofu-forget10:trained"]
        assert len(ds.points) == 4
        sens = _load(run / "sensitivity.json")
        assert len(sens) == 2
        assert (run / "plane.png").exists()

    def test_tofu_forget_quality_is_recorded_per_split(self, run):
        p = _load(run / "partial" / "alpha_1.000.json")
        assert set(p["tofu_forget_quality"]) == {"trained", "heldout", "all"}
        for v in p["tofu_forget_quality"].values():
            assert v is None or 0.0 <= v <= 1.0
        cal = _load(run / "calibration.json")
        assert "tofu_forget_quality_full_vs_retain90" in cal

    def test_reference_outcomes_support_fact_level_analysis(self, run):
        from deeperase.scripts.fact_level_analysis import analyse_run
        cal = _load(run / "calibration.json")
        p = _load(run / "partial" / "alpha_1.000.json")
        for split in ("full", "retain90"):
            assert set(cal["reference_outcomes"][split]) == set(p["breadth_tiers"]["outcomes"])
        r = analyse_run(run, n_boot=20)
        assert r["baseline"] == "full" and set(r["splits"]) == {"trained", "heldout"}
        assert [row["alpha"] for row in r["splits"]["trained"]["per_alpha"]] == [0.0, 1.0]

    def test_selectivity_is_breadth_minus_retain_loss(self, run):
        p = _load(run / "partial" / "alpha_1.000.json")
        for pt in p["points"]:
            assert pt["selectivity"] == pytest.approx(pt["breadth"] - pt["retain_loss"])

    def test_resume_reuses_every_saved_point(self, run, checkpoints):
        before = _load(run / "plane.json")
        for f in (run / "partial").glob("alpha_*.json"):
            f.touch()
        _run(run.parent, checkpoints, "--protocol", "v2", "--method", "ga",
             "--n-heldout-examples", "6", "--run-id", "e2e_v2", "--resume",
             "--breadth-items", str(ITEMS_V2))
        assert _load(run / "plane.json") == before
        # the resume retrained; on CPU that is bit-reproducible, and recorded
        assert _load(run / "unlearn.json")["resumed_from"]["relative_drift"] == \
            pytest.approx(0.0, abs=1e-9)


class TestRunDirectoryIntegrity:
    """A run directory must never end up holding points from two different runs."""

    def _cfg(self, **kw):
        base = {"method": "ga", "size": "1B", "protocol": "v2", "seed": 0,
                "unlearn": {"learning_rate": 1e-6}, "dtype": "float32",
                "inputs": {"breadth_items_sha256": "aa", "reference_spans_sha256": "bb"},
                "git": {"commit": "c1"}, "alphas": [0.0, 1.0]}
        base.update(kw)
        return base

    def test_resume_with_the_same_settings_records_the_resume(self, tmp_path):
        from deeperase.scripts.run_study import write_config
        p = tmp_path / "config.json"
        write_config(p, self._cfg(), resume=False)
        write_config(p, self._cfg(git={"commit": "c2"}, alphas=[0.0, 0.5, 1.0]), resume=True)
        saved = _load(p)
        assert saved["git"]["commit"] == "c1"              # the original is kept
        assert saved["resumes"][0]["git"]["commit"] == "c2"
        assert saved["resumes"][0]["alphas"] == [0.0, 0.5, 1.0]

    @pytest.mark.parametrize("change", [
        {"protocol": "v1"}, {"seed": 1}, {"unlearn": {"learning_rate": 3e-6}},
        {"dtype": "bfloat16"},
        {"inputs": {"breadth_items_sha256": "zz", "reference_spans_sha256": "bb"}}])
    def test_resume_with_different_settings_is_refused(self, tmp_path, change):
        from deeperase.scripts.run_study import write_config
        p = tmp_path / "config.json"
        write_config(p, self._cfg(), resume=False)
        with pytest.raises(SystemExit, match="--resume refused"):
            write_config(p, self._cfg(**change), resume=True)

    def test_resuming_a_pre_protocol_run_is_refused(self, tmp_path):
        """v1 directories were scored with the defective breadth scorer."""
        from deeperase.scripts.run_study import write_config
        p = tmp_path / "config.json"
        old = {k: v for k, v in self._cfg().items() if k != "protocol"}
        p.write_text(json.dumps(old))
        with pytest.raises(SystemExit, match="protocol"):
            write_config(p, self._cfg(protocol="v1"), resume=True)

    def test_reusing_a_run_id_without_resume_is_refused(self, tmp_path, checkpoints):
        from deeperase.scripts.run_study import main
        run = tmp_path / "taken"
        (run / "partial").mkdir(parents=True)
        (run / "partial" / "alpha_0.000.json").write_text("{}")
        with pytest.raises(SystemExit, match="already holds"):
            main(["--model-sources", json.dumps(checkpoints), "--output-dir", str(tmp_path),
                  "--run-id", "taken", "--device", "cpu"])


class TestProtocolV1GradDiff:
    @pytest.fixture(scope="class")
    def run(self, checkpoints, tmp_path_factory):
        out = tmp_path_factory.mktemp("v1")
        _run(out, checkpoints, "--protocol", "v1", "--method", "graddiff",
             "--run-id", "e2e_v1")
        return out / "e2e_v1"

    def test_v1_trains_on_exactly_the_depth_examples(self, run):
        s = _load(run / "split.json")
        rows = sorted(int(x.split("_")[1]) for x in s["depth_examples"]["trained"])
        assert s["trained_rows"] == rows and len(rows) == 6
        assert "heldout" not in s["depth_examples"]
        u = _load(run / "unlearn.json")
        assert u["history"]["n_steps"] == 1           # 6 rows, batch 8, 1 epoch

    def test_v1_design_and_retain_split(self, run):
        c = _load(run / "config.json")
        assert (c["forget_split"], c["unlearn"]["loss_scope"]) == ("depth", "full")
        assert c["retain_config"] == "retain90"
        assert c["inputs"]["breadth_items"].endswith("data/breadth_items.json")

    def test_v1_has_one_trajectory_on_the_pooled_headline(self, run):
        from deeperase.eval.plane import PlaneDataset
        ds = PlaneDataset.load(run / "plane.json")
        assert [t.benchmark for t in ds.trajectories()] == ["tofu-forget10"]
        cal = _load(run / "calibration.json")
        assert cal["headline_groups"] == ["forget:all"]
        p = _load(run / "partial" / "alpha_0.000.json")
        assert "RA" not in p["groups"]                # v1 item set has no RA/WF


class TestLrScan:
    """The scan must train exactly as the study does and keep every rate's curve."""

    def _scan(self, checkpoints, out, lrs):
        from deeperase.scripts.lr_scan import main
        argv = ["--method", "ga", "--lrs", lrs, "--epochs", "2", "--batch-size", "8",
                "--model-sources", json.dumps(checkpoints), "--tofu-dir", str(TOFU),
                "--cache-dir", str(out / "cache"), "--device", "cpu",
                "--n-breadth", "12", "--max-length", "256", "--out-dir", str(out / "scans"),
                "--breadth-items", str(ITEMS_V2)]
        assert main(argv) == 0
        return _load(out / "scans" / "ga_1B_v2_s0.json")

    def test_every_rate_gets_a_curve_and_a_verdict(self, checkpoints, tmp_path):
        d = self._scan(checkpoints, tmp_path, "1e-2,1e-5")
        assert set(d["rates"]) == {"0.01", "1e-05"}
        assert d["loss_scope"] == "answer" and d["dtype"] == "float32"
        for r in d["rates"].values():
            assert r["baseline_utility"] is not None and r["epochs"]
            assert r["verdict"] in ("too fast", "usable", "too slow")
            assert r["n_epochs_run"] <= 2

    def test_a_second_scan_adds_to_the_same_record(self, checkpoints, tmp_path):
        self._scan(checkpoints, tmp_path, "1e-5")
        d = self._scan(checkpoints, tmp_path, "3e-6")
        assert set(d["rates"]) == {"1e-05", "3e-06"}
