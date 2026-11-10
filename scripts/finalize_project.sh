#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Ensuring required packages are listed in requirements.txt (preserving existing lines)..."
touch requirements.txt
for pkg in numpy scikit-learn scipy matplotlib pytest; do
    if ! grep -qi "^${pkg}" requirements.txt; then
        echo "$pkg" >> requirements.txt
    fi
done

echo "[2/6] Appending Day 7 & Day 8 research summary to README.md (preserving existing README content)..."
if [ -f README.md ] && ! grep -q "Day 7 Controlled Degree-Heterogeneity Benchmark" README.md; then
    cp README.md README.md.bak
    cat README_DAY7_DAY8_SECTION.md >> README.md
    echo "Appended Day 7 & Day 8 documentation to existing README.md (backup saved at README.md.bak)."
fi

echo "[3/6] Running unit and mathematical invariant test suite..."
python3 -m unittest discover -s tests -v

echo "[4/6] Running Day 7, Day 8, and figure generation (existing results/ and figures/ are preserved)..."
mkdir -p results paper/figures
python3 -m experiments.degree_heterogeneity > results/degree_heterogeneity.txt
python3 -m experiments.regime_interaction_study > results/regime_interaction_study.txt
python3 -m experiments.generate_paper_figures

echo "[5/6] Compiling LaTeX research paper if pdflatex is available..."
if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    (
        cd paper
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        bibtex main
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        cp main.pdf ../spectral_graph_clustering_research_paper.pdf
        rm -f main.aux main.bbl main.blg main.log main.out
    )
    echo "Compiled spectral_graph_clustering_research_paper.pdf successfully."
else
    echo "pdflatex/bibtex not found locally; keeping pre-compiled spectral_graph_clustering_research_paper.pdf."
fi

echo "[6/6] Auditing git repository state, preserving figures/ and notebooks/, and pushing..."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    echo "Current branch: $CURRENT_BRANCH"
    git rm -r --cached --ignore-unmatch src/__pycache__ experiments/__pycache__ >/dev/null 2>&1 || true

    git add .gitignore requirements.txt README.md
    git add src/ experiments/ results/ tests/ scripts/finalize_project.sh
    git add paper/main.tex paper/references.bib paper/figures/*.pdf spectral_graph_clustering_research_paper.pdf
    [ -d figures ] && git add figures/
    [ -d notebooks ] && git add notebooks/

    git status --short
    if git diff --cached --quiet; then
        echo "No staged changes to commit."
    else
        git commit -m "Complete Day 7 & Day 8 degree heterogeneity study, tests, and research paper"
    fi

    git fetch origin "$CURRENT_BRANCH"
    git push origin "$CURRENT_BRANCH"
    echo "Successfully updated origin/$CURRENT_BRANCH."
else
    echo "Not inside a git work tree; skipping git commit/push."
fi
