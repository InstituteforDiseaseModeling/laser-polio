import matplotlib.pyplot as plt
import numpy as np

# # Add persistent vaccine access/uptake score (e.g., logit scale, or in [0,1])
# self.people.add_scalar_property("vaccination_propensity", dtype=np.float32)

# Example 1: Logistic-distributed access (centered around 0.5)
# Logit-normal is reasonable for probability modeling
n = 1000  # self.people.count
logit_center = 0.8
scale = 1.0
logits = np.random.normal(loc=np.log(logit_center / (1 - logit_center)), scale=scale, size=n)
props = 1 / (1 + np.exp(-logits))
# self.people.vaccination_propensity[:] = props

plt.figure(figsize=(8, 5))
plt.hist(props, bins=50, color="skyblue", edgecolor="black")
plt.xlim(0, 1)
plt.title("Distribution of Vaccination Propensity")
plt.xlabel("Vaccination Propensity")
plt.ylabel("Number of Individuals")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")
