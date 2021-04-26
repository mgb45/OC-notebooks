# OC-notebooks
A series of notebooks for model-based DRL or optimal control. Includes direct trajectory optimisation, CEM policy training and CEM planning.

Given running cost $g(x_t,u_t)$ and terminal cost $h(x_T)$ the finite horizon $(t=0 \ldots T)$ optimal control problem seeks to find the optimal control, 
$$u^*_{1:T} = \text{argmin}_{u_{1:T}} L(x_{1:T},u_{1:T})$$ 
$$u^*_{1:T} = \text{argmin}_{u_{1:T}} h(x_T) + \sum_{t=0}^T g(x_t,u_t)$$
subject to the dynamics constraint: $x_{t+1} = f(x_t,u_t)$.

These notebooks demonstrate solutions to problems of this form, using the inverted pendulum as an example, and assuming dynamics are not know a-priori. First, we gather state, actions, next state pairs, and use these to train a surrogate neural network dynamics model, $x_{t+1} \sim \hat{f}(x_t,u_t)$, approximating the true dynamics $f$. We'll then set up a suitable optimiser to find $$u^*$$ online by rolling out using the surrogate dynamics $\hat{f}$

Check out https://github.com/mgb45/ilqr_pendulum for an example using a linear Gaussian policy, and ILQR.
