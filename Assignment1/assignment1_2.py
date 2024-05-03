import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp

def f(x):
    """The function to integrate: exp(x)."""
    return np.exp(x)

def analytic_integral():
    """Compute the analytic solution of the integral of exp(x) from 0 to 1."""
    return np.exp(1) - np.exp(0)

def monte_carlo_integral(f, samples):
    """Estimate an integral using provided samples."""
    evaluations = f(samples)
    return np.mean(evaluations)

def relative_error(approx, ref):
    """Compute the relative error between the approximation and reference."""
    return abs(approx - ref) / ref

# Reference solution
ref_solution = analytic_integral()

# Sample sizes
sample_sizes = [10, 100, 1000, 10000, 100000]

# Distributions
uniform_dist = cp.Uniform(0, 1)

# Storing results for plotting
qmc_results = []
qmc_errors = []

# Perform Quasi-Monte Carlo Integration using Halton sequences for increasing sample sizes
for N in sample_sizes:
    samples = uniform_dist.sample(size=N, rule='H')
    qmc_approx = monte_carlo_integral(f, samples)
    qmc_results.append(qmc_approx)
    error = relative_error(qmc_approx, ref_solution)
    qmc_errors.append(error)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.loglog(sample_sizes, qmc_errors, marker='o', linestyle='-', color='r', label='QMC - Halton')
plt.title('Relative Error of QMC Integration vs. Sample Size')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Relative Error (log scale)')
plt.legend()
plt.grid(True)
plt.show()
    
