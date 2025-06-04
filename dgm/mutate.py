import copy
import random
import uuid


def mutate(config: dict) -> tuple[str, dict]:
    """Return a new agent id and mutated config."""
    new_config = copy.deepcopy(config)
    # Example hyper-params
    if "lr" in new_config:
        new_config["lr"] *= random.uniform(0.8, 1.2)
    if "lora_r" in new_config:
        delta = random.choice([-1, 1]) * random.randint(0, 2)
        new_config["lora_r"] = max(1, new_config["lora_r"] + delta)
    agent_id = str(uuid.uuid4())[:8]
    return agent_id, new_config

