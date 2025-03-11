import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# A demo of how to generate correlated samples from two distributions
# Based on Kurt's description in the EMOD_NGA_Model_Overview_2025_02_12.pdf

# Step 1: Define parameters for Lognormal
mean_lognormal = 1
variance_lognormal = 4

# Convert mean and variance to log-space parameters
mu_ln = np.log(mean_lognormal**2 / np.sqrt(variance_lognormal + mean_lognormal**2))
sigma_ln = np.sqrt(np.log(variance_lognormal / mean_lognormal**2 + 1))

# Step 2: Define parameters for Gamma
mean_gamma = 14 / 24
shape_gamma = 1  # Arbitrary shape parameter
scale_gamma = mean_gamma / shape_gamma  # scale = mean / shape

# Step 3: Generate correlated normal samples
n_samples = 10000  # Number of samples
rho = 0.8  # Desired correlation

# Create covariance matrix
cov_matrix = np.array([[1, rho], [rho, 1]])
L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition

# Generate standard normal samples
z = np.random.normal(size=(n_samples, 2))
z_corr = z @ L.T  # Apply Cholesky to introduce correlation

# Step 4: Transform normal variables into target distributions
lognormal_samples = np.exp(mu_ln + sigma_ln * z_corr[:, 0])  # Lognormal transformation
gamma_samples = stats.gamma.ppf(stats.norm.cdf(z_corr[:, 1]), a=shape_gamma, scale=scale_gamma)  # Gamma transformation

# Step 5: Verify correlation
corr_actual = np.corrcoef(lognormal_samples, gamma_samples)[0, 1]
print(f"Achieved correlation: {corr_actual:.4f}")

# Step 6: Plot results
plt.figure(figsize=(6, 6))
plt.scatter(lognormal_samples, gamma_samples, alpha=0.3, s=5)
plt.xlabel("Risk multiplier (lognormal samples)")
plt.ylabel("Daily infectivity (gamma samples)")
plt.title(f"Correlation between individual risk and infectivity (ρ ≈ {corr_actual:.2f})")
plt.show()
