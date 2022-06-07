## Agricultural extensification

This repository contains code resources to accompany our research paper,

>Variational inference for cutting feedback in misspecified models.

These codes are for the example in Section 6.2: *Agricultural extensification.*

In this example, we fit Bayesian full model and cut model by variational approximation and MCMC on the agriculture data.

Detailed steps to run the code are as follows.

- 1 Variational approximation

  + Step 1.1: Run [VB_cut_stage_one.py] and then run [VB_cut_stage_two.py] to get the variational posteriors for cut model.
  + Step 1.2: Run [VB_full_adam.py] to get the variational posteriors for full model.

- 2 MCMC and SMC

   + Step 2.1: Run [MCMC_cut_stage_one.py] and then run [MCMC_cut_stage_two.py] to get the MCMC posteriors for cut model.
   + Step 2.2: Run [SMC_initialization.py] and then run [SMC_full.py] to get the SMC posteriors for full model.

- 3 Generate figures

   + Step 3.1: Run [agri_plot.py]. It will save [gamma.pdf], [gamma_full_comp.pdf] as Figure 10 in the paper, [check1.pdf], [check2.pdf], [check3,pdf] as Figure 9 in the paper. 