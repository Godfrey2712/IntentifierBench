import os
from pathlib import Path
import shutil

PAN19_ROOT = Path("pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23")
OUT_ROOT = Path("documents_pan19")
PROBLEMS = [f"problem{idx:05d}" for idx in range(1, 6)]
UNKNOWN_KEEP = {f"unknown{idx:05d}.txt" for idx in range(1, 21)}  # unknown00001.txt ... unknown00020.txt

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def concat_candidate(candidate_dir: Path) -> str:
    parts = []
    for f in sorted(candidate_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".txt":
            parts.append(read_text(f))
    return "\n".join(parts).strip()

def process():
    OUT_ROOT.mkdir(exist_ok=True)
    for problem in PROBLEMS:
        problem_dir = PAN19_ROOT / problem
        if not problem_dir.is_dir():
            print(f"skip missing {problem_dir}")
            continue

        out_folder = OUT_ROOT / problem.replace("problem", "documents")
        out_folder.mkdir(parents=True, exist_ok=True)

        for candidate in sorted([d for d in problem_dir.iterdir() if d.is_dir() and d.name.startswith("candidate")]):
            text = concat_candidate(candidate)
            out_path = out_folder / f"{candidate.name}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"wrote candidate profile {out_path}")

        unknown_dir = problem_dir / "unknown"
        if unknown_dir.is_dir():
            for ufile in sorted(unknown_dir.iterdir()):
                if ufile.is_file() and ufile.name in UNKNOWN_KEEP:
                    target = out_folder / ufile.name
                    shutil.copy2(ufile, target)
                    print(f"copied unknown file {ufile} -> {target}")
        else:
            print(f"no unknown dir in {problem_dir}")

    print("done")

if __name__ == "__main__":
    process()