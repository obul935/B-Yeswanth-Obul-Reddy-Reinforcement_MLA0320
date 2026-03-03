import numpy as np

# Grid size
GRID_SIZE = 5

# Pickup location (example)
pickup_location = (4, 4)

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

# Initialize value function
V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)


def is_valid(state):
    x, y = state
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def get_reward(state):
    if state == pickup_location:
        return 20
    return -1


def value_iteration():
    global V
    while True:
        delta = 0
        new_V = np.copy(V)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                state = (i, j)

                if state == pickup_location:
                    continue

                action_values = []

                for a in actions:
                    dx, dy = actions[a]
                    next_state = (i + dx, j + dy)

                    if not is_valid(next_state):
                        next_state = state

                    reward = get_reward(next_state)
                    value = reward + gamma * V[next_state]
                    action_values.append(value)

                best_value = max(action_values)
                new_V[i, j] = best_value
                delta = max(delta, abs(best_value - V[i, j]))

        V = new_V

        if delta < theta:
            break


def extract_policy():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = (i, j)

            if state == pickup_location:
                policy[i, j] = -1
                continue

            action_values = []

            for a in actions:
                dx, dy = actions[a]
                next_state = (i + dx, j + dy)

                if not is_valid(next_state):
                    next_state = state

                reward = get_reward(next_state)
                value = reward + gamma * V[next_state]
                action_values.append(value)

            policy[i, j] = np.argmax(action_values)


value_iteration()
extract_policy()

print("Optimal Value Function:\n")
print(V)

print("\nOptimal Policy (0=Up,1=Down,2=Left,3=Right,-1=Pickup):\n")
print(policy)
