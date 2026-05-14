# Dreamer: Model-Based Reinforcement Learning

A PyTorch implementation of Dreamerv1 and Dreamerv2 for continuous control tasks using the DeepMind Control Suite.


https://github.com/user-attachments/assets/f4e4c5da-5366-40ff-91cb-e0d02d4e916e


---

## Overview

Dreamer learns a world model from pixel observations and trains an actor-critic policy entirely inside the model's imagination — making it significantly more sample efficient than model-free methods.

```
Real Environment → World Model → Imagination → Actor-Critic → Policy
```

---

## Project Structure

```
dreamer/
├── dreamer.py        # Main training loop + Dreamer class
├── models.py         # RSSM, ConvEncoder, ConvDecoder, ActionDecoder, DenseDecoder
├── buffer.py         # Replay buffer
├── env_wrapper.py    # DeepMind Control Suite wrappers
├── utils.py          # Logger, FreezeParameters, compute_return
└── README.md         # This file
```

---

## Architecture

### 1. World Model
Learns to understand the environment from pixels.

```
Pixels (3×64×64)
    ↓
ConvEncoder → embedding (1024 dims)
    ↓
RSSM → latent state (stoch=30 + deter=200 = 230 dims)
    ↓
ConvDecoder     → reconstructed pixels
RewardDecoder   → predicted reward
DiscountDecoder → predicted discount (Dreamerv2 only)
```

### 2. RSSM (Recurrent State Space Model)
The heart of Dreamer — models environment dynamics in latent space.

```
Two components:
  Deterministic (deter=200): GRU hidden state → memory
  Stochastic    (stoch=30):  sampled from Normal → uncertainty

Two distributions:
  Prior:     imagined next state (no observation)
  Posterior: corrected state     (with observation)

KL loss forces prior → match posterior
"Make imagination match reality"
```

### 3. Actor (ActionDecoder)
Trained entirely in imagination — never touches real environment during training.

```
Latent state (230 dims)
    ↓
4 × Linear(400) + ELU
    ↓
mean + std → TanhNormal distribution
    ↓
action (6 dims for walker-walk)
```

### 4. Critic (Value Model)
Estimates expected future reward from latent state.

```
Latent state (230 dims)
    ↓
3 × Linear(400) + ELU
    ↓
expected return (scalar)
```

---

## Training Pipeline

```
Step 1 — Collect Real Experience
  Random policy → replay buffer (seed_steps=5000)

Step 2 — Train World Model
  Sample sequences from buffer
  Encode observations → RSSM → decode
  Loss = obs_loss + reward_loss + kl_loss

Step 3 — Imagine Rollouts
  Start from real posterior states
  Actor takes actions for 15 dream steps
  World model predicts next states + rewards

Step 4 — Train Actor
  Maximize TD-λ returns in imagination
  actor_loss = -mean(discounts × returns)

Step 5 — Train Critic
  Predict TD-λ returns accurately
  value_loss = -mean(discounts × log_prob(returns))

Step 6 — Collect More Data
  Use trained actor to collect new experiences
  Repeat from Step 2
```

---

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd dreamer

# Install dependencies
pip install torch torchvision
pip install dm_control
pip install gym
pip install tensorboardX
pip install imageio
pip install numpy matplotlib
```

---

## Usage

### Training

```python
# In Colab or script
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args([])

# Key settings
args.env            = 'walker-walk'   # DeepMind Control task
args.algo           = 'Dreamerv1'     # or 'Dreamerv2'
args.train          = True
args.total_steps    = 200000          # training steps
args.seed_steps     = 5000            # random exploration steps
args.batch_size     = 50
args.train_seq_len  = 50
args.imagine_horizon= 15

# Run
main()
```

### Evaluation

```python
eval_rews, video_images = dreamer.evaluate(
    test_env,
    eval_episodes=10,
    render=True)

print(f"Mean reward: {np.mean(eval_rews):.2f}")
```

### Visualize Training

```python
# TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/data/

# Plot reward curve
import pickle
import matplotlib.pyplot as plt

logs = []
with open('scalar_data.pkl', 'rb') as f:
    while True:
        try:
            logs.append(pickle.load(f))
        except EOFError:
            break

steps   = [l['step'] for l in logs]
rewards = [l.get('train_avg_reward', 0) for l in logs]
plt.plot(steps, rewards)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Training Reward')
plt.show()
```

---

## Hyperparameters

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| stoch_size | 30 | Stochastic latent size |
| deter_size | 200 | GRU hidden size |
| obs_embed_size | 1024 | CNN encoder output size |
| num_units | 400 | Hidden units in MLP models |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| total_steps | 5,000,000 | Total environment steps |
| seed_steps | 5,000 | Random exploration steps |
| batch_size | 50 | Training batch size |
| train_seq_len | 50 | Sequence length for RSSM |
| imagine_horizon | 15 | Dream rollout length |
| update_steps | 100 | Gradient updates per iteration |
| collect_steps | 1,000 | Environment steps per iteration |

### Coefficients

| Parameter | Default | Description |
|-----------|---------|-------------|
| free_nats | 3.0 | KL loss lower bound |
| discount | 0.99 | Reward discount factor |
| td_lambda | 0.95 | TD-λ return mixing |
| kl_loss_coeff | 1.0 | KL loss weight |
| action_noise | 0.3 | Exploration noise |

### Optimizers

| Parameter | Default | Description |
|-----------|---------|-------------|
| model_learning_rate | 6e-4 | World model LR |
| actor_learning_rate | 8e-5 | Actor LR |
| value_learning_rate | 8e-5 | Critic LR |
| grad_clip_norm | 100.0 | Gradient clipping |

---

## Results

### Walker-Walk (50k steps)

| Step | Train Reward |
|------|-------------|
| 10,000 | ~20 |
| 20,000 | ~104 |
| 30,000 | ~222 |
| 50,000 | ~245 |
| Eval (50k) | ~302 |

### World Model Quality
Reconstruction quality after 50k steps:
- Shape accuracy: 9/10
- Color accuracy: 8/10
- Background: 9/10
- Overall: 8/10

---

## Key Concepts

### Why Dreamer is Sample Efficient
```
Model-free (PPO/SAC): needs ~10M real steps
Dreamer:             needs ~1M real steps (10× better)

Reason:
1 real step → update world model
World model → generate 1000s of imagined steps
Actor trains on imagined steps → very efficient
```

### KL Divergence
Forces the prior (imagination) to match the posterior (reality):
```
KL(posterior || prior) = 0   → perfect imagination
KL(posterior || prior) large → imagination wrong
Free nats = 3.0 prevents posterior collapse
```

### TD-Lambda Returns
Blends short and long horizon return estimates:
```
λ=0:   r + γV(s')           low variance, biased
λ=1:   r + γr' + γ²r'' ...  high variance, unbiased
λ=0.95: weighted mix         best of both worlds
```

---

## Dreamerv1 vs Dreamerv2

| Feature | Dreamerv1 | Dreamerv2 |
|---------|-----------|-----------|
| KL loss | Standard | Balanced (α=0.8) |
| Discount model | No | Yes |
| Stability | Good | Better |
| Sample efficiency | Good | Better |

Switch via:
```python
args.algo = 'Dreamerv2'
args.use_disc_model = True
args.kl_alpha = 0.8
```

---

## Extending the Project

### Add V-JEPA Encoder
Replace ConvEncoder with pretrained V-JEPA for richer visual features:
```python
from vjepa_encoder import VJEPAEncoder
self.obs_encoder = VJEPAEncoder(embed_size=1024)
```

### Add Imitation Learning
Train actor to mimic expert demonstrations:
```python
# Combined loss
actor_loss = dreamer_actor_loss + bc_coeff * imitation_loss
```

### Noise Annealing
Reduce exploration noise over time:
```python
noise = max(0.1, 0.3 * (1 - global_step / total_steps))
args.action_noise = noise
```

---

## File Descriptions

### `dreamer.py`
Main file containing the `Dreamer` class with:
- `_build_model()` — initializes all networks
- `world_model_loss()` — obs + reward + KL loss
- `actor_loss()` — imagination rollout + TD-λ returns
- `value_loss()` — critic training
- `train_one_batch()` — single training iteration
- `act_and_collect_data()` — collect with exploration
- `evaluate()` — test without exploration

### `models.py`
Neural network definitions:
- `RSSM` — recurrent state space model
- `ConvEncoder` — pixel → embedding
- `ConvDecoder` — embedding → pixel
- `DenseDecoder` — latent → reward/value/discount
- `ActionDecoder` — latent → action
- `TanhBijector` — action space transformation
- `SampleDist` — sampling distribution wrapper

### `buffer.py`
Experience replay:
- `ReplayBuffer` — stores and samples sequences

### `env_wrapper.py`
Environment wrappers:
- `DeepMindControl` — DM Control interface
- `ActionRepeat` — repeat actions n times
- `NormalizeActions` — scale actions to [-1, 1]
- `TimeLimit` — episode length limit

### `utils.py`
Utilities:
- `Logger` — TensorBoard + pickle logging
- `FreezeParameters` — context manager for gradients
- `compute_return` — TD-λ return computation

---

## References

- [Dream to Control (Dreamerv1)](https://arxiv.org/abs/1912.01603) — Hafner et al. 2019
- [Mastering Atari with Discrete World Models (Dreamerv2)](https://arxiv.org/abs/2010.02193) — Hafner et al. 2020
- [DeepMind Control Suite](https://arxiv.org/abs/1801.00690) — Tassa et al. 2018

---

## License

MIT License — free to use and modify.
