from pathlib import Path
from .archive import Archive
from .mutate import mutate
from .validate import evaluate


BEST_INFO_PATH = Path("outputs/dgm_best.json")


def save_best_info(agent_id: str, score: float, path: Path = BEST_INFO_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "{" + f"\"agent_id\": \"{agent_id}\", \"score\": {score}" + "}"
    )


def run(generations: int = 2, population: int = 2, archive_path: str = "outputs/dgm_archive.json"):
    """Run a tiny DGM loop."""
    base_cfg = {"lr": 1e-4, "lora_r": 16}
    archive = Archive(archive_path)

    for _ in range(generations):
        for _ in range(population):
            agent_id, cfg = mutate(base_cfg)
            score = evaluate(agent_id, cfg)
            archive.add(agent_id, score, cfg)

    archive.save()
    best_id, best_score = archive.get_best()
    if best_id is not None:
        save_best_info(best_id, best_score, BEST_INFO_PATH)
    print(f"[DGM] Best agent {best_id} score {best_score:.3f}")
    return best_id, best_score

