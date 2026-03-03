import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of ads (arms)
n_ads = 5

# True click probabilities for each ad (unknown to algorithm)
true_ctr = np.array([0.05, 0.08, 0.12, 0.04, 0.10])

n_rounds = 10000


# Simulate user click (environment)
def simulate_click(ad):
    return 1 if np.random.rand() < true_ctr[ad] else 0


# ------------------ EPSILON GREEDY ------------------
def epsilon_greedy(epsilon=0.1):
    counts = np.zeros(n_ads)
    values = np.zeros(n_ads)
    total_clicks = 0
    ctr_history = []

    for t in range(n_rounds):
        if np.random.rand() < epsilon:
            ad = np.random.randint(n_ads)
        else:
            ad = np.argmax(values)

        reward = simulate_click(ad)
        total_clicks += reward

        counts[ad] += 1
        values[ad] += (reward - values[ad]) / counts[ad]

        ctr_history.append(total_clicks / (t + 1))

    return ctr_history


# ------------------ UCB ------------------
def ucb():
    counts = np.zeros(n_ads)
    values = np.zeros(n_ads)
    total_clicks = 0
    ctr_history = []

    for t in range(n_rounds):
        if 0 in counts:
            ad = np.argmin(counts)
        else:
            ucb_values = values + np.sqrt(2 * np.log(t) / counts)
            ad = np.argmax(ucb_values)

        reward = simulate_click(ad)
        total_clicks += reward

        counts[ad] += 1
        values[ad] += (reward - values[ad]) / counts[ad]

        ctr_history.append(total_clicks / (t + 1))

    return ctr_history


# ------------------ THOMPSON SAMPLING ------------------
def thompson_sampling():
    successes = np.zeros(n_ads)
    failures = np.zeros(n_ads)
    total_clicks = 0
    ctr_history = []

    for t in range(n_rounds):
        sampled_theta = np.random.beta(successes + 1, failures + 1)
        ad = np.argmax(sampled_theta)

        reward = simulate_click(ad)
        total_clicks += reward

        if reward == 1:
            successes[ad] += 1
        else:
            failures[ad] += 1

        ctr_history.append(total_clicks / (t + 1))

    return ctr_history


# Run algorithms
eg_ctr = epsilon_greedy()
ucb_ctr = ucb()
ts_ctr = thompson_sampling()

# Plot CTR comparison
plt.figure(figsize=(10, 6))
plt.plot(eg_ctr, label="Epsilon-Greedy")
plt.plot(ucb_ctr, label="UCB")
plt.plot(ts_ctr, label="Thompson Sampling")
plt.xlabel("Rounds")
plt.ylabel("Click Through Rate (CTR)")
plt.title("Comparison of Bandit Algorithms for Ad Selection")
plt.legend()
plt.show()

# Print final CTR
print("Final CTR:")
print("Epsilon-Greedy:", eg_ctr[-1])
print("UCB:", ucb_ctr[-1])
print("Thompson Sampling:", ts_ctr[-1])
