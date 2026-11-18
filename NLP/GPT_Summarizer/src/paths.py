from pathlib import Path

def project_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "configs" / "config.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (configs/config.json).")
