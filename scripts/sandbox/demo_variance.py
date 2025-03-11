import matplotlib.pyplot as plt
import numpy as np

# Given parameters
R0 = 14.0  # Basic reproduction number
c = 10  # Avg. contacts per day
D = 5  # Avg. infectious period (days)
ind_variance = 4.0  # Variance in individual risk
corr = 0.8  # Correlation between acquisition and transmission risk

# Compute baseline transmission probability per contact
beta_0 = R0 / (c * D)

# Compute log-normal variance and covariance matrix
sigma = np.sqrt(np.log(1 + ind_variance))  # Log-normal std deviation
cov_matrix = np.array([[sigma**2, corr * sigma**2], [corr * sigma**2, sigma**2]])

# Generate correlated individual risk multipliers (log-space)
log_risks = np.random.multivariate_normal([0, 0], cov_matrix, size=1000)

# Convert from log-space to real-space
acq_risk, trans_risk = np.exp(log_risks[:, 0]), np.exp(log_risks[:, 1])

# Compute individual transmission probabilities
beta_individuals = beta_0 * acq_risk * trans_risk

# Plot acq_risk vs trans_risk

plt.figure(figsize=(10, 6))
plt.scatter(acq_risk, trans_risk, alpha=0.5)
plt.show()

# Check if population-level mean beta produces desired R0
effective_R0 = np.mean(beta_individuals) * c * D
print(f"Empirical R0 from model: {effective_R0:.2f}")

# Scale beta_0 if needed to match target R0
scaling_factor = R0 / effective_R0
beta_individuals *= scaling_factor
