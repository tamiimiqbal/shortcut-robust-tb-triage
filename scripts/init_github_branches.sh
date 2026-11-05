#!/usr/bin/env bash
set -e
git init
git checkout -b main
git config user.name "Md Tamim Iqbal, Sujana Mahmuda Nite & Puja Saha"
git config user.email "22201141@uap-bd.edu"
git add README.md requirements.txt configs/
git commit -m "chore: initialize CSE 414 TB Triage project structure and config"
git checkout -b feature/data
git add data/
git commit -m "feat(data): implement CODA TB loader, 0.5s log-mel/MFCC dsp, patient & LOCO splits, and site-balanced weights"
git checkout main
git merge --no-ff feature/data -m "Merge pull request #1 from feature/data (Weeks 1-2 milestone)"
git checkout -b feature/models
git add models/
git commit -m "feat(models): implement 5 baselines, 2D CNN-BiLSTM, Clinical MLP, Gated Cross-Modal Attention, GRL Site Head, and TempScaling"
git checkout main
git merge --no-ff feature/models -m "Merge pull request #2 from feature/models (Weeks 3-6 milestone)"
git checkout -b feature/evaluation
git add evaluation/ run_pipeline.py results/
git commit -m "feat(evaluation): add LOCO benchmark, WHO 90%-sens triage metrics, country linear probe, shortcut slope, ECE, and ablations"
git checkout main
git merge --no-ff feature/evaluation -m "Merge pull request #3 from feature/evaluation (Weeks 7-8 milestone)"
git checkout -b feature/demo
git add demo/ notebooks/
git commit -m "feat(demo): add interactive Gradio TB triage web app and end-to-end Google Colab notebook"
git checkout main
git merge --no-ff feature/demo -m "Merge pull request #4 from feature/demo (Weeks 9-10 milestone)"
echo "Git repository initialized with main and 4 merged module branches!"
