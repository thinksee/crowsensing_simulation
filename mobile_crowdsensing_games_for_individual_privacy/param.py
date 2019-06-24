import numpy as np
# constant about model
DATA_RANGE = 10
CONFIDENCE_LEVEL = 0.95
PAYMENT_ACC = 0.5

USER_COST = np.arange(0.1, 1.1, 0.1)
USER_ACTION = np.arange(0.1, 1.1, 0.1)
MCS_ACTION = np.arange(0.1, 0.55, 0.02)

N_USER_MULTI = 60
N_USER_SINGLE = 2

INTERVAL = 100  # for the experiment

FUNC = 1  # only 1(reciprocal) and 2(percentage)

ALGO = 2  # 1(q-learning) 2(cnn-dqn)

POLICY = 2  # 1(e-greedy) 2 (random) 3 (greedy)

# some constant about learning algorithm.
# about cnn
MEMORY_CAPACITY = 1000
WINDOW = 15
BATCH_SIZE = 32
ANNEAL_STEP = 300
GAMMA = 0.5
EPS_START = 1.0
EPS_END = 0.1
LEARNING_BEGIN = 10

# about experiment
MAX_EPISODE = 10
MAX_STEP = 50



