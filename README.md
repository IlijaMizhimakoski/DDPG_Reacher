# DDPG Implementations for Solving Reacher-v5

This repository contains two complete implementations of the **Deep Deterministic Policy Gradient (DDPG)** algorithm, both capable of solving the Reacher-v5 continuous-control environment from Gymnasium. The two versions differ in architecture, normalization, and initialization strategy, which leads to different learning behavior and performance profiles.

## 1. Paper-Accurate DDPG Implementation

This version closely follows the original DDPG paper.

**Key characteristics:**

- Uses LayerNorm in the actor and critic networks
- Applies fan-in weight initialization for all layers
- Critic network structure:
    - The state is processed in the first layer
    - In the second layer, the action is concatenated with the state output and forwarded together
- Exhibits more stable training
- Requires more episodes to fully learn the policy, likely due to smaller/controlled weight initialization slowing early improvements

## 2. Simplified DDPG Implementation

This version removes complexity and uses a more standard architecture.

**Key characteristics:**

- No LayerNorm
- No special weight initialization
- The critic receives state and action concatenated from the very first layer
- Learns faster
- Achieved the best-performing model in ~5000 episodes

Both implementations use **Ornstein–Uhlenbeck (OU) noise** for exploration and share the same training loop structure, replay buffer logic, and evaluation procedure. Although both approaches solve the environment, the simplified implementation converged faster.

The **hyperparameters** used in both implementations were inspired by the original DDPG paper, with some tweaks to improve learning speed and stability for the Reacher-v5 environment.

---
**DDPG model (simplified implementation) trained for 5000 episodes.**

![ddpgreacher](https://github.com/user-attachments/assets/6e66852b-1640-49ab-b4dd-e933953bd8ce)

<img width="700" height="360" alt="best_model_reward" src="https://github.com/user-attachments/assets/b3534c55-f6b2-4a18-9d78-890811757f6d" />

---

*Note: The GIF above is somewhat choppy due to format limitations. For a smoother and full-quality version, check out the [YouTube video here](https://youtu.be/poUWi_RIz9o?si=5sPu-vIG-smee3Nk).*
