
# Console
SEED_PRINT_FREQ = 1000
COLLECT_PRINT_FREQ = 250
TRAIN_PRINT_FREQ = 1
EVAL_PRINT_FREQ = 125

# Data
BATCH_SIZE = 32
WINDOW_SIZE = 32
NUM_ASSETS = 32
INDICATORS = True

# Training
NUM_EPOCHS = 1000
SEED_EPOCHS = 0
UPDATE_STEPS = 200

# Experience Replay Buffer
BUFFER_SIZE = 425000
BUFFER_MIN_SIZE = 1000
INCLUDE_LAST = True

# Optimization
ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
ALPHA_LR = 0.0003

# Agent
GAMMA = 0.997
TAU = 0.005
TAU_B = 0.005
DELAY_UPDATE = 2
CLIPPING_RANGE = 3
GRAD_BIAS = 0.1
STD_BIAS = 0.1

# Actor
HIDDEN_DIM = 256
MIN_LOG_STD = -20
MAX_LOG_STD = 0.5

# Policy
POLICY = "LSRE-CANN"

if POLICY == "LSRE-CANN":
    DEPTH = 1
    NUM_LATENTS = 12
    LATENT_DIM = 128
    NUM_CROSS_HEADS = 4
    CROSS_HEAD_DIM = 64
    NUM_SELF_HEADS = 4
    SELF_HEAD_DIM = 64
    DROPOUT = 0.1
