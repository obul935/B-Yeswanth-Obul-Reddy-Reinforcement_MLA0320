import numpy as np
import random

grid_size = 5
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
gamma = 0.9

dirt_cells = {(0,2), (1,4), (3,1), (4,3)}
obstacles = {(1,1), (2,3), (3,3)}

def step(state, action):
    r, c = state
    if action == 'UP':
        r -= 1
    elif action == 'DOWN':
        r += 1
    elif action == 'LEFT':
        c -= 1
    elif action == 'RIGHT':
        c += 1
    
    if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
        return state, -0.1
    
    next_state = (r, c)
    
    if next_state in obstacles:
        return state, -1
    
    reward = 0
    if next_state in dirt_cells:
        reward = 1
    
    return next_state, reward

def random_policy(state):
    return random.choice(actions)

def greedy_policy(state, dirt):
    if not dirt:
        return random.choice(actions)
    sr, sc = state
    target = min(dirt, key=lambda d: abs(d[0]-sr)+abs(d[1]-sc))
    tr, tc = target
    if tr < sr:
        return 'UP'
    if tr > sr:
        return 'DOWN'
    if tc < sc:
        return 'LEFT'
    if tc > sc:
        return 'RIGHT'
    return random.choice(actions)

def value_iteration():
    V = np.zeros((grid_size, grid_size))
    policy = np.full((grid_size, grid_size), '', dtype=object)
    
    for _ in range(100):
        new_V = np.copy(V)
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) in obstacles:
                    continue
                values = []
                for a in actions:
                    (nr, nc), reward = step((r,c), a)
                    values.append(reward + gamma * V[nr][nc])
                new_V[r][c] = max(values)
                policy[r][c] = actions[np.argmax(values)]
        V = new_V
    return policy

def simulate(policy_type='random', max_steps=50):
    state = (0,0)
    dirt = set(dirt_cells)
    total_reward = 0
    optimal_policy = value_iteration()
    
    print("Start State:", state)
    
    for step_count in range(max_steps):
        if not dirt:
            break
        
        if policy_type == 'random':
            action = random_policy(state)
        elif policy_type == 'greedy':
            action = greedy_policy(state, dirt)
        elif policy_type == 'optimal':
            action = optimal_policy[state]
        
        next_state, reward = step(state, action)
        
        if next_state in dirt:
            dirt.remove(next_state)
        
        total_reward += reward
        state = next_state
        
        print(f"Step {step_count+1}: Action={action}, State={state}, Reward={reward}")
    
    print("Total Reward:", total_reward)
    print("Remaining Dirt:", dirt)

print("\nRandom Policy Simulation")
simulate('random')

print("\nGreedy Policy Simulation")
simulate('greedy')

print("\nOptimal Policy Simulation")
simulate('optimal')
