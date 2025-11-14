!pip install numpyro

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pymc.sampling.jax as pmjax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

histogram_control = {ADD YOUR HISTOGRAM HERE}
histogram_treatment = {ADD YOUR HISTOGRAM HERE}

def aggregated_data(experiment_histogram):

  size = sum(experiment_histogram.values())
  # Number of converters
  converters = sum(v for k, v in experiment_histogram.items() if k > 0)
  # Number of conversions
  conversions = sum(k * v for k, v in experiment_histogram.items())
  # Variance of the distribution:
  # Compute mean of counts
  mean = conversions / size

  # Compute variance = E[X^2] - (E[X])^2
  E_x2 = sum((k**2) * v for k, v in experiment_histogram.items()) / size
  variance = E_x2 - mean**2

  return size, converters, conversions, mean, variance

n_C, c_C, x_C, mean_C, var_C = aggregated_data(histogram_control)
n_T, c_T, x_T, mean_T, var_T = aggregated_data(histogram_treatment)

# normal distribution model
delta = mean_T - mean_C
se_diff = np.sqrt(var_C / n_C + var_T / n_T)
samples_normal = np.random.normal(loc=delta, scale=se_diff, size=100000)

# two-step aggregate Bayesian model
with pm.Model() as model:
    # Priors for conversion rate
    p_T = pm.Beta("p_T", alpha=1, beta=1)
    p_C = pm.Beta("p_C", alpha=1, beta=1)

    # Priors for conversion intensity (conversions per converter)
    lambda_T = pm.Gamma("lambda_T", alpha=2, beta=1)
    lambda_C = pm.Gamma("lambda_C", alpha=2, beta=1)

    # Observed converters
    conv_T = pm.Binomial("conv_T", n=n_T, p=p_T, observed=c_T)
    conv_C = pm.Binomial("conv_C", n=n_C, p=p_C, observed=c_C)

    # Observed total conversions
    totalconv_T = pm.Poisson("totalconv_T", mu=lambda_T * c_T, observed=x_T)
    totalconv_C = pm.Poisson("totalconv_C", mu=lambda_C * c_C, observed=x_C)

    # Derived expected convs per user
    expected_T = pm.Deterministic("expected_T", p_T * lambda_T)
    expected_C = pm.Deterministic("expected_C", p_C * lambda_C)
    lift = pm.Deterministic("lift", expected_T - expected_C)

    trace_two_stage = pmjax.sample_numpyro_nuts(draws=50000, tune=3000, chains = 4, target_accept=0.97)

az.summary(trace_two_stage, var_names=["expected_T", "expected_C", "lift"])

lift_two_stage = trace_two_stage.posterior["lift"].values.flatten()

# Compute 5% and 95% credible interval
lower, upper = np.percentile(lift_two_stage, [5, 95])
print('Lift is: ', (trace_two_stage.posterior["lift"].values).mean())
print(f"90% credible interval for lift: [{lower:.6f}, {upper:.6f}]")


def prepare_counts(histogram):
    max_k = max(histogram.keys())
    counts_array = np.zeros(max_k + 1, dtype=int)
    for k, v in histogram.items():
        counts_array[k] = v
    return jnp.array(counts_array)

# 2. Dirichlet-Multinomial model for one group
def dirichlet_model(counts):
    k_vals = jnp.arange(len(counts))
    total_users = counts.sum()
    empirical_probs = counts / total_users

    log_c = numpyro.sample("log_c", dist.Normal(1.0, 1.0))
    c = jnp.exp(log_c)
    alpha = empirical_probs * c + 1e-3

    p = numpyro.sample("p", dist.Dirichlet(alpha))
    numpyro.sample("obs", dist.Multinomial(total_users, p), obs=counts)

    rate = jnp.dot(p, k_vals)
    numpyro.deterministic("rate", rate)
    numpyro.deterministic("concentration", c)

# 3. Run inference separately for each group
def run_model(histogram, seed=0):
    counts = prepare_counts(histogram)
    kernel = NUTS(dirichlet_model)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=20000)
    mcmc.run(random.PRNGKey(seed), counts=counts)
    return mcmc.get_samples()

dir_multinom_treatment = run_model(histogram_treatment, seed=0)
dir_multinom_control = run_model(histogram_control, seed=1)

# 6. Compute lift
lift_dir_multinom = dir_multinom_treatment["rate"] - dir_multinom_control["rate"]

# 7. Summarize
def summarize(name, samples):
    mean = samples.mean()
    lower = jnp.percentile(samples, 2.5)
    upper = jnp.percentile(samples, 97.5)
    print(f"{name}: mean = {mean:.6f}, 95% CI = [{lower:.6f}, {upper:.6f}]")

summarize("Treatment Rate", dir_multinom_treatment["rate"])
summarize("Control Rate", dir_multinom_control["rate"])
summarize("Lift (T - C)", lift_dir_multinom )


# Dirichlet-Poisson Mixture Model
def build_model(conversion_counts, frequencies, converter_rate, K=3):

    counts = np.asarray(conversion_counts, dtype='int32').reshape(-1)  # shape (N,)
    freqs = np.asarray(frequencies, dtype='int32').reshape(-1)    # shape (N,)

    with pm.Model() as model:
        w = pm.Dirichlet("mix_weights", a=np.ones(K, dtype="float32"))
        lambda_k = pm.Gamma("lambda_k", alpha=1.0, beta=1.0, shape=K)

        # We want the loglikelihood for each (count, lambda_k) pair
        # So build a broadcasted version of the Poisson logpmf, shape (N, K)
        # Use PyMC math, and rely on broadcasting.
        # counts: shape (N,). lambda_k: shape (K,)
        # The trick: let PyMC broadcast counts to (N, K) against lambda_k
        # Use pm.logp directly; input shapes will auto-broadcast.
        comp_logp = pm.logp(pm.Poisson.dist(mu=lambda_k), counts[:, None])
        log_w = pm.math.log(w)
        log_probs = log_w[None, :] + comp_logp

        mix_LL = pm.math.logsumexp(log_probs, axis=1)

        pm.Potential("likelihood", pm.math.sum(freqs * mix_LL))

        expected_lambda = pm.math.dot(w, lambda_k)
        pm.Deterministic("expected_lambda_converters", expected_lambda)
        population_rate = converter_rate * expected_lambda
        pm.Deterministic("population_conversion_rate", population_rate)
        prob_converter = 1.0 - pm.math.dot(w, pm.math.exp(-lambda_k))
        pm.Deterministic("prob_converter", prob_converter)

    return model

converter_rate_control = c_C / n_C
conversion_counts_control = np.asarray(list(histogram_control.keys()))
frequencies_control = np.asarray(list(histogram_control.values()))

K_param = 3
model = build_model(conversion_counts_control, frequencies_control, converter_rate_control, K = K_param)
dir_poisson_control = pmjax.sample_numpyro_nuts(draws=50000, tune=5000, chains=2, target_accept=0.8, random_seed=42, model=model,)

# --- Summarize Posterior ---
summary = az.summary(dir_poisson_control, var_names=["population_conversion_rate", "expected_lambda_converters", "prob_converter"])
print(summary)

# Optional: compare to empirical rate
empirical_rate = sum(k * v for k, v in histogram_control.items()) / n_C
print(f"\nEmpirical mean conversion rate: {empirical_rate:.8f}")

converter_rate_treatment = c_T / n_T
conversion_counts_treatment = np.asarray(list(histogram_treatment.keys()))
frequencies_treatment = np.asarray(list(histogram_treatment.values()))

model = build_model(conversion_counts_treatment, frequencies_treatment, converter_rate_treatment, K = K_param)
dir_poisson_treatment = pmjax.sample_numpyro_nuts(draws=50000, tune=5000, chains=2, target_accept=0.8, random_seed=42, model=model,)

# --- Summarize Posterior ---
summary = az.summary(dir_poisson_treatment, var_names=["population_conversion_rate", "expected_lambda_converters", "prob_converter"])
print(summary)

# Optional: compare to empirical rate
empirical_rate = sum(k * v for k, v in histogram_treatment.items()) / n_T
print(f"\nEmpirical mean conversion rate: {empirical_rate:.8f}")

lift_dir_pois = dir_poisson_treatment.posterior["expected_lambda_converters"].values.flatten() - dir_poisson_control.posterior["expected_lambda_converters"].values.flatten()
