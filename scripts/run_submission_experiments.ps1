param(
    [string]$CasesFile = "data/evaluation_cases_extended.csv"
)

$ErrorActionPreference = "Stop"

Write-Host "Running GraphRAG-Gene submission experiments..." -ForegroundColor Cyan
Write-Host "Using evaluation cases: $CasesFile" -ForegroundColor Cyan

python src/build_index.py --data_dir data/raw --output_dir results/index --skip_llm
python src/run_baseline_experiment.py --index_dir results/index --cases_file $CasesFile --output_dir results/tables --match_mode synonym
python src/evaluate_resolution.py --data_dir data/raw --output_dir results/tables
python src/run_ablation_experiment.py --index_dir results/index --cases_file $CasesFile --output_dir results/tables --match_mode synonym
python src/run_baseline_experiment.py --index_dir results/index --cases_file $CasesFile --output_dir results/tables_strict --match_mode strict
python src/run_ablation_experiment.py --index_dir results/index --cases_file $CasesFile --output_dir results/tables_strict --match_mode strict
python src/export_case_reports.py --index_dir results/index --output_dir results/cases --offline
python src/generate_manuscript_figures.py --tables_dir results/tables --index_dir results/index --output_dir results/figures
python src/generate_system_diagrams.py

Write-Host "Done. Check results/tables, results/tables_strict, results/cases, and results/figures." -ForegroundColor Green
