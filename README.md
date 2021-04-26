# OC-notebooks
A series of notebooks for model-based DRL or optimal control. Includes direct trajectory optimisation, CEM policy training and CEM planning.

Given running cost g and terminal cost h the finite horizon optimal control problem seeks to find the optimal control that mimimises these, subject to a dynamics constraint.

These notebooks demonstrate solutions to problems of this form, using the inverted pendulum as an example, and assuming dynamics are not know a-priori. First, we gather state, actions, next state pairs, and use these to train a surrogate neural network dynamics model, approximating the true dynamics $f$. We'll then set up a suitable optimiser to find the optimal control online (or a policy that produces these) by rolling out using the surrogate dynamics model.

Check out https://github.com/mgb45/ilqr_pendulum for an example using a linear Gaussian policy, and ILQR.
