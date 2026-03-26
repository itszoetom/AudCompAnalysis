# Discriminability

Pairwise stimulus discriminability analyses - future pass.

Planned scripts:
- `discriminability_analysis.py` - shared functions (Pearson correlation, linear SVM)
- `plot_discriminability_pearson.py` - pairwise Pearson correlation heatmaps and boxplots
- `plot_discriminability_svm.py` - pairwise linear SVM accuracy heatmaps and boxplots

Datasets: Natural Sounds Dataset (A1/AuD/AuV/AuP), Speech Dataset (A1/AuV/AuP)
Methods: Pearson correlation (n=5 sessions, 100 subsamplings, 30 neurons/session);
         Linear SVM with leave-one-out CV; Mann-Whitney U + Bonferroni correction
