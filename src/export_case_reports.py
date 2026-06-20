import argparse
from pathlib import Path

from query_gene import query

DEFAULT_CASES = {
    "tumor_susceptibility_HRR": "BRCA2 PALB2 CHEK2",
    "mitochondrial_respiration": "SDHA NDUFS4 SURF1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="results/index")
    parser.add_argument("--output_dir", default="results/cases")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, question in DEFAULT_CASES.items():
        genes, answer = query(args.index_dir, question, offline=args.offline, max_pathways=12, include_community=False)
        path = out / f"{name}.txt"
        path.write_text(answer, encoding="utf-8")
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
