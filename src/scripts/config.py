import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(path: str) -> dict:
    cfg = load_yaml(path)
    inherit = cfg.get("inherit", None)
    if inherit:
        base = load_yaml(inherit)
        # shallow merge (good enough for skeleton)
        base.update(cfg)
        cfg = base
    return cfg
