
# Console
ROLLOUT_PRINT_FREQ = 250
TRAIN_PRINT_FREQ = 1
EVAL_PRINT_FREQ = 125

# Data
BATCH_SIZE = 64

# Training
NUM_EPOCHS = 1000
SEED_EPOCHS = 0
UPDATE_STEPS = 200
EPSILON = 0.1

# Optimization
LR = 0.001

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
else:
    raise NotImplementedError(f"Policy {POLICY} not implemented")