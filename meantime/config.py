import os

#### STRINGS ####
# STATE DICT
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
SCHEDULER_STATE_DICT_KEY = 'scheduler_state_dict'
TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY = 'train_loader_dataset_rng_state'
TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY = 'train_loader_sampler_rng_state'
STEPS_DICT_KEY = 'steps'

RECENT_STATE_DICT_FILENAME = 'recent_checkpoint.pth'
BEST_STATE_DICT_FILENAME = 'best_model.pth'

# STATUS STRINGS
STATUS_RUNNING = 'running\n'
STATUS_FINISHED = 'finished\n'
STATUS_RECOVERY = 'recovery\n'

#### CONFIGS ####
# WANDB
USE_WANDB = False

# REMOTE-MACHINE-RELATED
MACHINE_IS_HOST = True
HOST, PORT, USERNAME, PASSWORD = None, None, None, None
REMOTE_ROOT = None

# DATA FOLDER
LOCAL_DATA_FOLDER = './Data'
