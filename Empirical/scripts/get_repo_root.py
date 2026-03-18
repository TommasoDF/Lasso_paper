from pathlib import Path

def get_repo_root():
    """Finds the repository root containing 'Empirical' and 'Theory' folders."""
    current = Path.cwd().resolve()
    repo_root = next((p for p in [current, *current.parents] if (p / 'Empirical').exists() and (p / 'Theory').exists()), None)
    if repo_root is None:
        raise FileNotFoundError("Could not locate repository root containing 'Empirical' and 'Theory'.")
    return repo_root