import numpy as np

# MCS
AGENT_MCS_ACTIONS = np.arange(0, 51, 2)
CONTRIBUTION_FACTOR = np.arange(5, 16, 1)

# Vehicular
V_ACTIONS = np.arange(0, 11, 1)
SNR_SET = [1, 10]
SNR_PROB_SET = [0.1, 0.9]
V_GAMMA = 0.9
MAX_SPEED = 5
V_SPEED_SET = np.arange(0, 6, 1)
V_SPEED_PROB_SET = [0.02, 0.04, 0.3, 0.3, 0.3, 0.04]
V_COST_SET = np.arange(0, 5.5, 0.5)

BETA = 20

PAYMENT_ACC = 0.5
# policy: policy=1, e-greedy; policy=2, random; policy=3, greedy
POLICY = 1

# about experiment
MAX_EPISODE = 6
MAX_STEP = 20
