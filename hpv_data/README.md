##Human pappilomavirus and cervical cancer incidence

This repository contains code resources to accompany our research paper,

>Variational inference for cutting feedback in misspecified models.

These codes are for the example in Section 6.1: *Human pappilomavirus and cervical cancer incidence*.

In this example, we fit Bayesian full model and cut model by variational approximation and MCMC on an HPV dataset.

These codes contain two parts, R codes and Python codes. R codes are used to get MCMC results, while Python codes are used to get variational approximation results.

Detailed steps to run the code are as follows.

- 1 MCMC

  + Step 1.1: Run R code [hpv-MCMC.R] and [hpv-MCMC-cut] to get MCMC posterior samples for full model and cut model respectively. It will save 2 data files: [hpv-full.csv] and [hpv-cut.csv] to your workspace.
  + Step 1.2: Copy [hpv-full.csv] and [hpv-cut.csv] generated from step 1.1 to the [Variational_approximation] folder for further results visualization.

- 2 Variational approximation

   + Step 2.1: Run Python code [hpv_full.py] and [hpv_cut.py] to get varitaional posteriors for full model and cut model respectively.
   + Step 2.2: Run Python code [hpv_plot.py]. It will save [HPV_JOINT2.pdf] as Figure 6 in the paper.
   + Step 2.3: Run Python code [hpv_conflict_check.py]. It will save [HPV-conflict-check-broken.pdf] as Figure 5 in the paper.






