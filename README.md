# GraphRAG-Gene

A Reactome-based knowledge graph retrieval framework for pathway-level interpretation of multi-gene queries.

GraphRAG-Gene constructs a heterogeneous gene-pathway graph, applies Leiden community detection, and performs specificity-aware pathway retrieval. The repository contains the reproducible offline workflow, a 36-case evaluation benchmark, baseline and ablation experiments, and publication figures.

## Authors

Zheng Wu and Yunqing Liu  
Luoyang Institute of Science and Technology

School of Computer Science, Luoyang Institute of Science and Technology, Luoyang 471000, China.

## 1. Data
Place the following files in `data/raw/`:

- `graphrag_entities.csv`
- `graphrag_relationships.csv`
- `genes.csv`
- `pathways.csv`
- `participates_in.csv`
- `interacts_with.csv`

Raw Reactome-derived tables are not committed to this repository. Place authorized copies of the required files in `data/raw/`.

## 2. Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, if `python` is not available, install Python 3.11+ and make sure it is on PATH before running the commands above.

## 3. Environment variables

Create a `.env` file if LLM summarization is needed:

```bash
ONEAPI_BASE_URL=http://localhost:3000/v1
ONEAPI_API_KEY=your_key_here
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v1
```

Never commit API keys to a public repository.

## 4. Build the index

```bash
python src/build_index.py --data_dir data/raw --output_dir results/index --skip_llm
```

Use `--skip_llm` for reproducible offline testing. Remove it when the API is configured.

## 5. Query examples

```bash
python src/query_gene.py --index_dir results/index --question "BRCA2 PALB2 CHEK2"
python src/query_gene.py --index_dir results/index --question "SDHA NDUFS4 SURF1"
```

For offline reproducibility without LLM calls:

```bash
python src/query_gene.py --index_dir results/index --question "BRCA2 PALB2 CHEK2" --offline
```

## 6. Evaluation

Run the baseline comparison:

```bash
python src/run_baseline_experiment.py --index_dir results/index --cases_file data/evaluation_cases.csv --output_dir results/tables
```

Run the Leiden resolution sweep:

```bash
python src/evaluate_resolution.py --data_dir data/raw --output_dir results/tables
```

Run the ablation comparison:

```bash
python src/run_ablation_experiment.py --index_dir results/index --cases_file data/evaluation_cases.csv --output_dir results/tables
```

Generate manuscript figures:

```bash
python src/generate_manuscript_figures.py --tables_dir results/tables --index_dir results/index --output_dir results/figures
```

Or run the full offline experiment workflow on Windows PowerShell:

```powershell
.\scripts\run_submission_experiments.ps1
```

The workflow uses `data/evaluation_cases_extended.csv` by default. To reproduce the original 12-case benchmark:

```powershell
.\scripts\run_submission_experiments.ps1 -CasesFile data/evaluation_cases.csv
```

By default, `results/tables` uses synonym-aware keyword matching for expected pathway labels, while `results/tables_strict` stores the strict substring-matching control.

## 7. Generate descriptive statistics

The index-building script writes:

- `results/index/create_final_nodes.parquet`
- `results/index/create_final_relationships.parquet`
- `results/index/create_final_communities.parquet`
- `results/index/kg_statistics.csv`
- `results/index/community_statistics.csv`

These tables can be used in the manuscript.

## 8. Manuscript positioning

This project should be described as a pathway-level systems biology interpretation framework. It should not be described as a clinical diagnostic system, personal disease risk predictor, or treatment recommendation system.

The framework is intended for pathway-level systems biology research. It is not a clinical diagnostic system, personal disease-risk predictor, variant pathogenicity assessor, or treatment recommendation system.

## Repository

<https://github.com/rainlove1101/GraphRAG-Gene>
