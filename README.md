# Variational Inference for Cutting Feedback in Misspecified Models

This repository contains code resources to accompany our research paper **"Variational inference for cutting feedback in misspecified models"**. The code implements variational inference approaches for cutting feedback in Bayesian models and reproduces all experimental results from the paper.

## Repository Structure

```
├── Biased_Data_Example/            # Code for Section 3.4 & 5.3 experiments
├── HPV_Example/                    # Code for Section 6.1 experiments
│   ├── MCMC/                       # R code for MCMC implementation
│   └── Variational_approximation/  # Python code for variational approximation
├── Agriculture_Example/            
│   ├── code/                       # Code for Section 6.2 experiments
│   └── MCMCsamples/                # Code for visualizing MCMC samples 
└── README.md                       # This documentation
```

## Reproducing Experimental Results

### 1. Biased Data Example (Sections 3.4 & 5.3)

This section reproduces Figures 2-4 from the paper.

**Implementation Steps:**

1. Run `biased_data.py`:
   - Generates `biased_marginal1.pdf` (Figure 2, left panel)
   - Generates `biased_marginal2.pdf` (Figure 2, right panel) 
   - Generates `biased_contour.pdf` (Figure 3)

2. Run `biased_data_conflict_check.py`:
   - Generates `biased-data-conflict-check.pdf` (Figure 4)

### 2. Human Papillomavirus and Cervical Cancer Incidence (Section 6.1)

This section reproduces Figures 5-6 from the paper, comparing variational approximation with MCMC.

#### MCMC Implementation (R code)

1. Run `hpv-MCMC.R` to obtain full model posterior samples (`hpv-full.csv`)
2. Run `hpv-MCMC-cut.R` to obtain cut model posterior samples (`hpv-cut.csv`)
3. Copy both CSV files to the `Variational_approximation` folder

#### Variational Approximation (Python code)

1. Run `hpv_full.py` for full model variational posterior
2. Run `hpv_cut.py` for cut model variational posterior
3. Run `hpv_plot.py` to generate `HPV_JOINT2.pdf` (Figure 6)
4. Run `hpv_conflict_check.py` to generate `HPV-conflict-check-broken.pdf` (Figure 5)

### 3. Agricultural Extensification (Section 6.2)

This section reproduces Figures 9-10 from the paper and provides a quick visualization of MCMC samples.

#### Reproduce Experiment Result
##### Variational Approximation

1. Run `VB_cut_stage_one.py` followed by `VB_cut_stage_two.py` for the cut model
2. Run `VB_full_adam.py` for the full model

##### MCMC and SMC Implementation

1. Run `MCMC_cut_stage_one.py` followed by `MCMC_cut_stage_two.py` for the cut model
2. Run `SMC_initialization.py` followed by `SMC_full.py` for the full model

##### Generate Figures

Run `agri_plot.py` to generate:
- `gamma.pdf` and `gamma_full_comp.pdf` (Figure 10)
- `check1.pdf`, `check2.pdf`, `check3.pdf` (Figure 9)

#### Quick Visualization of MCMC Samples

1. Run `MCMCsamples/read_MCMC_samples.py` 


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yu2023variational,
  title={Variational inference for cutting feedback in misspecified models},
  author={Yu, Xuejun and Nott, David J and Smith, Michael Stanley},
  journal={Statistical Science},
  volume={38},
  number={3},
  pages={490--509},
  year={2023},
  publisher={Institute of Mathematical Statistics}
}
```
