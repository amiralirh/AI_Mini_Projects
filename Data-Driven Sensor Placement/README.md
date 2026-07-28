# Data-Driven Sensor Placement for Non-Linear Trajectories in Barrier Coverage Systems

This repository contains the simulation code and models for an extended approach to the sensor placement problem in 2D barrier coverage systems. 

The base framework for this problem traditionally assumes that targets travel in straight lines, using a log-Gaussian Cox line process (LGCLP) to model traffic. This project relaxes that strict geometric assumption by introducing a data-driven approach using applied machine learning. Specifically, a sequence-to-sequence Variational Autoencoder (VAE) is implemented to learn actual, non-linear movement patterns from trajectory data and map them into a probabilistic latent space for near-optimal sensor deployment.

## Problem Description

The base paper approaches the sensor placement problem for barrier coverage by assuming that targets move in straight lines. To handle this, it uses a bijective transformation to map 2D linear trajectories from the physical space to distinct points in a representation space, denoted by $\mathcal{C} = (\alpha, p)$. The target traffic is then estimated in this new space. 

However, assuming targets only travel in straight lines is a noticeable limitation for real-world systems. In scenarios like marine navigation or autonomous monitoring, targets naturally follow non-linear paths to steer around obstacles or execute control maneuvers. If we approximate these curved paths as straight lines over large areas, the geometric model loses accuracy, which ultimately results in suboptimal sensor placements.

**Proposed Extension:** 
To address this limitation, this project replaces the standard geometric representation with a Variational Autoencoder (VAE). Instead of relying on assumed straight lines, the VAE learns actual, non-linear movement patterns directly from historical data and maps them into a probabilistic latent space. This allows the model to drop the strict linear assumption and handle the natural uncertainty of real-world target behavior much more reliably.

## Repository Structure

* `data_generation.py`: Script to generate synthetic non-linear trajectory datasets with incorporated measurement noise and sinusoidal perturbations.
* `vae_model.py`: Contains the TensorFlow/Keras architecture for the LSTM-based encoder, custom sampling layer (reparameterization trick), and decoder.
* `train.py`: The main training loop that minimizes the Evidence Lower Bound (ELBO) loss (Reconstruction MSE + KL Divergence).
* `optimization.py`: Downstream pipeline that evaluates sensor spatial performance against the generated latent space distributions to calculate void probability.
* `visualization/`: Contains plotting scripts for training convergence, trajectory reconstruction, and generative sampling.

## Methodology & Simulation Setup

1. **Synthetic Data Generation:** Due to the unavailability of raw AIS datasets, the model is trained on a synthetic dataset of 2,000 non-linear trajectories. These sequences include randomized base velocities, obstacle-avoidance maneuvers, and Gaussian measurement noise to replicate real-world uncertainty.
2. **VAE Architecture:** A sequence-to-sequence VAE processes the temporal trajectory data. The encoder utilizes two stacked LSTM layers to compress the sequences into a 4-dimensional continuous latent space. The decoder expands this sampled vector back across the time steps to project it into the 2D physical space.
3. **Training Protocol:** The network is trained using the Adam optimizer. The custom loss function forces the network to act as both a non-linear state estimator (filtering high-frequency noise) and a generative model (ensuring stable latent space distributions).
4. **Sensor Optimization:** The decoder generates new non-linear trajectories by sampling from the standard normal latent space. These samples estimate the target density function, replacing the linear Cox process intensity function, and are evaluated against spatial sensor models to determine final sensor coordinates.


### Prerequisites
Ensure you have the following dependencies installed:
* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib

