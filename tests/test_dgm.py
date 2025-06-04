import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dgm.archive import Archive
from dgm.mutate import mutate
from dgm.validate import evaluate
from dgm import loop


def test_archive_add_and_best(tmp_path):
    path = tmp_path / "archive.json"
    arch = Archive(path)
    arch.add("a1", 0.1, {})
    arch.add("a2", 0.5, {})
    arch.save()

    arch2 = Archive(path)
    assert arch2.get_best() == ("a2", 0.5)


def test_mutate_changes_params():
    cfg = {"lr": 1.0, "lora_r": 16}
    agent_id, new_cfg = mutate(cfg)
    assert agent_id
    # Likely mutated
    assert new_cfg != cfg


def test_validate_bounds():
    score = evaluate("agent", {})
    assert 0.0 <= score <= 1.0


def test_loop_runs(tmp_path):
    arch_path = tmp_path / "archive.json"
    best_path = tmp_path / "best.json"
    loop.BEST_INFO_PATH = best_path
    best_id, best_score = loop.run(generations=1, population=1, archive_path=str(arch_path))
    assert arch_path.exists()
    data = json.loads(arch_path.read_text())
    assert data["best_id"] == best_id
    assert best_path.exists()
