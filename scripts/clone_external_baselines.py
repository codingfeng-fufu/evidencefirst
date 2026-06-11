import argparse
import subprocess
from pathlib import Path


REPOS = {
    "arag": "https://github.com/Ayanami0730/arag",
    "ircot": "https://github.com/StonyBrookNLP/ircot",
    "hoprag": "https://github.com/LIU-Hao-2002/HopRAG",
}

BLOCKED_REPOS = {
    "catrag": "Official repository currently contains prompts/data only and says logic code will be released later.",
    "corag": "No confirmed official GitHub repo for Chain-of-Retrieval CoRAG; do not clone unrelated same-name projects.",
}

def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("external_repos"))
    parser.add_argument("--repo", choices=sorted(set(REPOS) | set(BLOCKED_REPOS)) + ["all"], default="all")
    parser.add_argument("--pull", action="store_true", help="Pull existing repositories instead of leaving them untouched")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    names = sorted(set(REPOS) | set(BLOCKED_REPOS)) if args.repo == "all" else [args.repo]
    failures = []

    for name in names:
        if name in BLOCKED_REPOS:
            print(f"blocked: {name}: {BLOCKED_REPOS[name]}")
            continue
        if name not in REPOS:
            failures.append(name)
            print(f"unknown repo: {name}")
            continue
        dest = args.root / name
        if dest.exists():
            if args.pull:
                rc = run(["git", "pull", "--ff-only"], cwd=dest)
                if rc:
                    failures.append(name)
            else:
                print(f"exists: {dest}")
            continue
        rc = run(["git", "clone", REPOS[name], str(dest)])
        if rc:
            failures.append(name)

    if failures:
        raise SystemExit(f"failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
