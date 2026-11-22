import os
import sys

sys.path.append('src')

from common.data.data_preprocessors import BaseDataPreprocessor

# Base directories
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAVEMASKS_RAW = r'C:\TA2\BRAVE-MASK\shared'
DIVINA_PROCESSED = r'C:\TA2\Dataset\DIVINA_processed'

data_config: dict = {
    'AGENDER': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
        },
    },
    'TIMIT': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
        },
    },
    'COMMONVOICE': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
        },
    },
    'VOXCELEB2': {
        'RAW_DATA_ROOT': '',
        'PROCESSED_DATA_ROOT': '',
        'VAD_FILE_PATH': '',
        'LABELS_FILE_PATH': '',
        'C_NAMES': {
            'gen': ['female', 'male'],
            'mask': ['No mask', 'Tissue mask', 'Medical mask', 
                     'Protective mask (ffp2/ffp3)', 'Respirator', 'Protective face shield', 'Hijab', 'Helmet', 'Glasses', 'Hat'],
        },
    },
    'BRAVEMASKS': {
        'RAW_DATA_ROOT': BRAVEMASKS_RAW,
        'PROCESSED_DATA_ROOT': BRAVEMASKS_RAW,
        'VAD_FILE_PATH': os.path.join(BASE_PROJECT_DIR, 'data', 'bravemasks_vad.pickle'),
        'LABELS_FILE_PATH': os.path.join(BASE_PROJECT_DIR, 'data', 'bravemasks_labels.csv'),
        'C_NAMES': {
            'gen': ['female', 'male'],
            'mask': ['No mask', 'Tissue mask', 'Medical mask', 'Protective mask (ffp2/ffp3)',
                     'Respirator', 'Protective face shield', 'Hijab', 'Helmet', 'Glasses', 'Hat'],
        },
    },
    'DIVINA': {
        'RAW_DATA_ROOT': DIVINA_PROCESSED,
        'PROCESSED_DATA_ROOT': DIVINA_PROCESSED,
        'VAD_FILE_PATH': os.path.join(BASE_PROJECT_DIR, 'data', 'divina_vad.pickle'),
        'LABELS_FILE_PATH': os.path.join(BASE_PROJECT_DIR, 'data', 'divina_labels.csv'),
        'C_NAMES': {
            'gen': ['female', 'male'],
            'mask': ['No mask', 'Tissue mask', 'Medical mask', 'Protective mask (ffp2/ffp3)',
                     'Respirator', 'Protective face shield', 'Hijab', 'Helmet', 'Glasses', 'Hat'],
        },
    },
}

training_config: dict = {
    'LOGS_ROOT': os.path.join(BASE_PROJECT_DIR, 'logs'),
    'MODEL': {
        'cls': None,
        'args': { 
        }
    },
    'FEATURE_EXTRACTOR': {
        'FEATURES_ROOT': os.path.join(BASE_PROJECT_DIR, 'data', 'features'),
        'FEATURES_FILE_NAME': 'SAMPLES',
        'WIN_MAX_LENGTH': 4,
        'WIN_SHIFT': 2,
        'WIN_MIN_LENGTH': 1,
        'SR': 16000,
    },
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 16,
    'OPTIMIZER': {'cls': 'AdamW', 'lr': 1e-4},
    'SCHEDULER': {'cls': 'CosineAnnealingWarmRestarts', 'T_0': 10, 'T_mult': 2},
    'AUGMENTATION': True,
    'USE_AMP': True,  # Mixed Precision Training (torch.cuda.amp), prevents CUDA OOM on 4GB VRAM
}
