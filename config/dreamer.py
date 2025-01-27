
# Training
NUM_EPOCHS = 500
BATCH_DIM = 16
BATCH_LEN = 64
BATCH_RATIO = 0.3

# Optimization
ACTOR_LR = 4e-5
CRITIC_LR = 4e-5
MODEL_LR = 4e-5
EPSILON = 1e-20
CLIP_GRAD = 0.3

# Two-Hot Encoding
LOW = -20
HIGH = 20
NUM_BINS = 199

'''--------------------------------| World Model |--------------------------------'''
PRED_SCALE = 1
DYN_SCALE = 1
REP_SCALE = 0.1
FREE_NATS = 1

RECURRENT_DIM = 512
STOCHASTIC_DIM = 32
DISCRETE_DIM = 32
UNIMIX = 0.01

# Sequence Model
SEQ_DIM = 512

# Embedding Model
NUM_HEADS = 4
DROPOUT = 0.1
NUM_ATTN_LAYERS = 2

EMBEDDING_NET = {"hidden_layers": [512],  "activation": "SiLU",  "normalize": True,  "init_zeros": False}

# Encoder Model
ENCODER_NET = {"hidden_layers": [512, 512, 512], "activation": "SiLU", "normalize": True, "init_zeros": False}

# Dynamics Model
DYNAMICS_NET = {"hidden_layers": [512, 512], "activation": "SiLU", "normalize": True, "init_zeros": False}

# Reward Model
REWARD_NET = {"hidden_layers": [512], "activation": "SiLU", "normalize": True, "init_zeros": True}

# Decoder Model
DECODER_NET = {"hidden_layers": [512, 512, 512], "activation": "SiLU", "normalize": True, "init_zeros": False}

'''--------------------------------| Agent |--------------------------------'''
HORIZON = 15
GAMMA = 0.997
LAMBDA = 0.95

# Actor
ACTOR_NET = {"hidden_layers": [512, 512, 512], "activation": "SiLU", "normalize": True, "init_zeros": False}

LOW = 0.0
HIGH = 1.0
NUM_BINS = 99
ACTION_REPEAT = 2

LOSS_SCALE = 1.0
ENTROPY_REG = 3e-4
UNIMIX = 0.0

RETNORM_DECAY = 0.01
RETNORM_LIMIT = 1.0
RETNORM_LOW = 0.05
RETNORM_HIGH = 0.95

RETNORM_EMA = {"decay": 0.01, "limit": 1.0, "low": 0.05, "high": 0.95}
ACTOR_EMA = {"decay": 0.01, "limit": 1e-8, "low": 0.0, "high": 1.0}

# Critic
CRITIC_NET = {"hidden_layers": [512, 512, 512], "activation": "SiLU", "normalize": True, "init_zeros": True}

LOSS_SCALE = 1.0
REPLAY_LOSS_SCALE = 0.3

CRITIC_EMA = {"decay": 0.01, "limit": 1e-8, "low": 0.0, "high": 1.0}

SLOW_REG = 1.0

TARGET_UPDATE_INTERVAL = 1
TARGET_FRAC = 0.02
