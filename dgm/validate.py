import random


def evaluate(agent_id: str, config: dict) -> float:
    """Placeholder evaluation returning a random score."""
    random.seed(hash(agent_id) % 2**32)
    return random.random()

