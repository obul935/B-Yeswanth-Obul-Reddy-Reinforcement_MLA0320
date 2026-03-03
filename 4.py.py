import numpy as np

# Grid size
GRID_SIZE = 6

# Multiple delivery locations
delivery_points = [(0, 5), (5, 5), (3, 2)]

# MDP parameters
gamma = 0.9
theta = 0.0001

# Actions: Up, Down, Left, Right
actions = {
    0: (-1, 0),   # Up
    1: (1, 0),    # Down
    2: (0, -1),   # Left
    3: (0, 1)     # Right
}

# Initialize value function and random policy
V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.random.randint(0, 4, size=(GRID_SIZE, GRID_SIZE))


def is_valid(state):
    x, y = state
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def get_reward(state):
    if state in delivery_points:
        return 50
    return -1


# ---------------- POLICY EVALUATION ----------------
def policy_evaluation():
    global V
    while True:
        delta = 0
        new_V = np.copy(V)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                state = (i, j)

                if state in delivery_points:
                    continue

                action = policy[i, j]
                dx, dy = actions[action]
                next_state = (i + dx, j + dy)

                if not is_valid(next_state):
                    next_state = state

                reward = get_reward(next_state)
                value = reward + gamma * V[next_state]
                new_V[i, j] = value

                delta = max(delta, abs(value - V[i, j]))

        V = new_V

        if delta < theta:
            break


# ---------------- POLICY IMPROVEMENT ----------------
def policy_improvement():
    global policy
    policy_stable = True

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = (i, j)

            if state in delivery_points:
                continue

            old_action = policy[i, j]
            action_values = []

            for a in actions:
                dx, dy = actions[a]
                next_state = (i + dx, j + dy)

                if not is_valid(next_state):
                    next_state = state

                reward = get_reward(next_state)
                value = reward + gamma * V[next_state]
                action_values.append(value)

            best_action = np.argmax(action_values)
            policy[i, j] = best_action

            if old_action != best_action:
                policy_stable = False

    return policy_stable


# ---------------- POLICY ITERATION ----------------
def policy_iteration():
    while True:
        policy_evaluation()
        if policy_improvement():
            break


policy_iteration()

print("Optimal Value Function:\n")
print(V)

print("\nOptimal Policy (0=Up,1=Down,2=Left,3=Right):\n")
print(policy)
