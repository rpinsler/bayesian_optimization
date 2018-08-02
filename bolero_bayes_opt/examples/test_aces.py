import random
import numpy as np
from scipy.stats import uniform

from bolero.environment.catapult import Catapult
from bolero_bayes_opt import SurrogateACESOptimizer
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

random.seed(0)
np.random.seed(0)

catapult = Catapult([(0, 0), (2.0, -0.5), (3.0, 0.5), (4, 0.25),
                     (5, 2.5), (7, 0), (10, 0.5), (15, 0)])
catapult.init()

def catapult_reward(params, target):
    catapult.request_context((target - 2.0) / 8.0)
    catapult.set_inputs(params)
    catapult.step_action()
    return catapult.get_feedback()[0]

n_rollouts = 25
evaluation_frequency = 5  # how often to evaluate learned policy
verbose = True
num_targets = 100
contexts = np.linspace(2, 10, num_targets)

samples = np.empty((n_rollouts, 3))
rewards = np.empty((n_rollouts))
offline_eval = np.empty((n_rollouts // evaluation_frequency, contexts.shape[0]))

kernel = C(100.0, (1.0, 10000.0)) \
    * RBF(length_scale=(1.0, 1.0, 1.0), length_scale_bounds=[(0.1, 100), (0.1, 100), (0.1, 100)])

opt = SurrogateACESOptimizer(context_boundaries=[(2, 8)], boundaries=[(5, 10), (0, np.pi/2)],
                    acquisition_function="EntropySearch", n_context_samples=10, kappa=1.)
opt.init(2, 1)

params_ = np.zeros(2)
reward = np.empty(1)
for rollout in range(n_rollouts):
    context = opt.get_desired_context()
    if context is None:
        context = uniform.rvs(2, 8, size=1)
    opt.set_context(context)

    opt.get_next_parameters(params_)
    samples[rollout] = (context[0], params_[0], params_[1])

    reward = catapult_reward(params_, context)

    rewards[rollout] = reward
    opt.set_evaluation_feedback(reward)

    if verbose:
        print("Rollout %d: Context: %.3f Velocity %.3f Angle %.3f Reward %.3f" %
              (rollout, context[0], params_[0], params_[1], reward))

    if (rollout + 1) % evaluation_frequency == 0:
        pol = opt.best_policy()
        offline_eval[rollout // evaluation_frequency] = \
            [catapult_reward(pol(np.array(c, ndmin=1), explore=False), c) for c in contexts]  # - best_rewards
        if verbose:
            print("Rollout %d: Average regret of policy %.3f" %
                  (rollout, offline_eval[rollout // evaluation_frequency].mean()))