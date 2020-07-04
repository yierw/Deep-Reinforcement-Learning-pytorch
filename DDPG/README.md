#  Deep Deterministic Policy Gradient (DDPG)

Implement DDPG [1] algorithm.

* `buffer.py`: replay buffer (the same as used in DQN)
* `model.py`: define actor and critic network
* `OUNoise.py`: Ornstein-Uhlenbeck process
* `ddpg.py`: ddpg agent
```python
# ---------------------------- update critic ---------------------------- #
next_actions = self.target_actor(next_states)
Q_next = self.target_critic(next_states, next_actions)
Q_targets = rewards + self.gamma * Q_next * (1. -dones)
Q_expected = self.critic(states, actions)
critic_loss = self.loss_fn(Q_expected, Q_targets.detach())
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
self.critic_optimizer.step()
# ---------------------------- update actor ---------------------------- #
pred_actions = self.actor(states)
actor_loss = -self.critic(states, pred_actions).mean()
self.actor_optimizer.zero_grad()
actor_loss.backward()
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
self.actor_optimizer.step()
# ---------------------------- update target net ---------------------------- #
soft_update(self.target_critic, self.critic, self.tau)
soft_update(self.target_actor, self.actor, self.tau)
```


Training tips:
* Huber loss is more efficient than MSE loss.
* Add noise sampled from noise process to enable exploration to this deterministic policy. The noise process can be chosen to suit environment. Ornstein-Uhlenbeck process is often used to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.

## References
[1] T.  P.  Lillicrap,  J.  J.  Hunt,  A.  Pritzel,  N.  Heess,  T.  Erez,  Y.  Tassa,  D.  Silver,  and  D.  Wierstra.Continuous control with deep reinforcement learning.arXiv preprint arXiv:1509.02971, 2015.
