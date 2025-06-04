import json
from pathlib import Path


class Archive:
    """Simple archive tracking DGM agents and best score."""

    def __init__(self, path: str = "outputs/dgm_archive.json"):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self.agents = data.get("agents", {})
            self.best_id = data.get("best_id")
            self.best_score = data.get("best_score", 0.0)
        else:
            self.agents = {}
            self.best_id = None
            self.best_score = 0.0

    def add(self, agent_id: str, score: float, config: dict):
        self.agents[agent_id] = {"score": score, "config": config}
        if score > self.best_score:
            self.best_score = score
            self.best_id = agent_id

    def get_best(self):
        return self.best_id, self.best_score

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {
                    "agents": self.agents,
                    "best_id": self.best_id,
                    "best_score": self.best_score,
                },
                f,
            )

