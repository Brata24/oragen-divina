import sys

sys.path.append('src')

import os
import pprint
import pickle
import datetime
from copy import deepcopy

import pandas as pd

import torch
import torch.nn as nn

from torchvision import transforms
from transformers import AutoConfig

from configs.multimodal_bm_config import data_config as dconf
from configs.multimodal_bm_config import training_config as tconf

from fusion.data.agender_multimodal_features_dataset import AGenderMultimodalFeaturesDataset

from fusion.features.common import FeaturesType

from common.data.grouping import mask_grouping
from common.data.common import define_context_length, DatasetType

from fusion.models.multimodal_models import MaskAgenderAVModelV1, MaskAgenderAVModelV2, MaskAgenderAVModelV3

from fusion.augmentation.modality_augmentation import ModalityDropAugmentation

from common.loss.loss import MaskAgenderLoss, compute_inverse_frequency_weights

from common.utils.accuracy import *

from common.net_trainer.multitask_net_trainer import MultiTaskNetTrainer

from common.utils.common import get_source_code, define_seed, AttrDict, wait_for_it
from fusion.run_eval_zeroshot_divina import ensure_configs


# Experiment scenarios for occlusion (mask) type recognition
SCENARIOS = {
    'A': {'corpora': ['BRAVEMASKS'], 'mask_num_classes': 6},            # Baseline: NM, TM, MM, PM, PFS, R
    'B': {'corpora': ['BRAVEMASKS', 'DIVINA'], 'mask_num_classes': 10}, # Extended: + HJ, HM, KA, TP
}
  

def main(d_config: dict, t_config: dict) -> None:
    """Trains with configuration in the following steps:
    - Defines datasets names
    - Defines data augmentations
    - Defines data preprocessor
    - Defines datasets
    - Defines dataloaders
    - Defines measures
    - Defines NetTrainer
    - Defines model
    - Defines weighted loss, optimizer, scheduler
    - Runs NetTrainer 

    Args:
        d_config (dict): Data configuration dictionary
        t_config (dict): Training configuration dictionary
    """
    # Defining class names
    corpora_names = list(d_config.keys())
    c_names = deepcopy(d_config[corpora_names[0]]['C_NAMES'])
    
    # Dynamic number of occlusion classes: 6 (Scenario A) or 10 (Scenario B)
    mask_num_classes = t_config.get('MASK_NUM_CLASSES', 6)
    mask_types = list(range(mask_num_classes))
    c_names['mask'] = c_names['mask'][:mask_num_classes]
    use_amp = t_config.get('USE_AMP', False)

    logs_root = t_config['LOGS_ROOT']
        
    features_root = t_config['FEATURE_EXTRACTOR']['FEATURES_ROOT']
    features_file_name = t_config['FEATURE_EXTRACTOR']['FEATURES_FILE_NAME']
    win_max_length = t_config['FEATURE_EXTRACTOR']['WIN_MAX_LENGTH']
    win_shift = t_config['FEATURE_EXTRACTOR']['WIN_SHIFT']
    win_min_length = t_config['FEATURE_EXTRACTOR']['WIN_MIN_LENGTH']
    features_type = t_config['FEATURE_EXTRACTOR']['FEATURES_TYPE']
    sr = t_config['FEATURE_EXTRACTOR']['SR']
        
    model_cls = t_config['MODEL']['cls']
    model_args = t_config['MODEL']['args']
    
    num_epochs = t_config['NUM_EPOCHS']
    batch_size = t_config['BATCH_SIZE']
    augmentation = t_config['AUGMENTATION']
    
    source_code = 'Data configuration:\n{0}\nTraining configuration:\n{1}\n\nSource code:\n{2}'.format(
        pprint.pformat(d_config),
        pprint.pformat(t_config),
        get_source_code([main, model_cls, AGenderMultimodalFeaturesDataset, MultiTaskNetTrainer]))
    
    # Defining datasets 
    ds_names = {
        'train': 'train',
        'devel': 'devel',
        'test': 'test',
    }
    
    c_names_to_display = {}
    for task, class_names in c_names.items():
        c_names_to_display[task] = [cn.capitalize() for cn in class_names]
    
    # Defining metadata
    metadata_info = {}
    for corpus_name in corpora_names:
        corpus_labels_metadata = pd.read_csv(d_config[corpus_name]['LABELS_FILE_PATH'], low_memory=False)
        with open(d_config[corpus_name]['VAD_FILE_PATH'], 'rb') as handle:
            corpus_vad_metadata = pickle.load(handle)
        
        has_devel = corpus_labels_metadata['subset'].isin(['devel', 'dev']).any()
        metadata_info[corpus_name] = {}
        for ds in ds_names:
            if 'train' in ds:
                # For DIVINA (has_devel=True), keep 'devel' separate from 'train' to prevent data leakage
                condition_list = corpus_labels_metadata['subset'].isin(['train']) if has_devel else corpus_labels_metadata['subset'].isin(['train', 'devel', 'dev'])
            elif 'devel' in ds:
                if not has_devel:
                    continue
                condition_list = corpus_labels_metadata['subset'].isin(['devel', 'dev'])
            else:
                condition_list = corpus_labels_metadata['subset'].isin(['test'])

            metadata_info[corpus_name][ds] = {
                'data_root': d_config[corpus_name]['PROCESSED_DATA_ROOT'],
                'labels_metadata': corpus_labels_metadata[condition_list],
                'features_file_name': '{0}_{1}_{2}'.format(corpus_name, ds_names[ds].upper(), features_file_name),
                'vad_metadata': corpus_vad_metadata
            }
    
    # Defining datasets
    datasets = {}
    datasets_stats = {}
    for corpus_name in corpora_names:
        datasets[corpus_name] = {}
        datasets_stats[corpus_name] = {}

        for ds in ds_names:
            if ds not in metadata_info[corpus_name]:
                continue
            datasets[corpus_name][ds] = AGenderMultimodalFeaturesDataset(data_root=metadata_info[corpus_name][ds]['data_root'],
                                                                         labels_metadata=metadata_info[corpus_name][ds]['labels_metadata'], 
                                                                         features_root=features_root,
                                                                         features_file_name=metadata_info[corpus_name][ds]['features_file_name'],
                                                                         corpus_name=corpus_name,
                                                                         gender_num_classes=len(c_names['gen']),
                                                                         features_type=features_type,
                                                                         channels=['c'],
                                                                         mask_types=mask_types,
                                                                         vad_metadata=metadata_info[corpus_name][ds]['vad_metadata'],
                                                                         sr=sr, win_max_length=win_max_length, win_shift=win_shift, 
                                                                         win_min_length=win_min_length,
                                                                         include_mask=True,
                                                                         transform=ModalityDropAugmentation() if ('train' in ds) and (augmentation) else None)

            datasets_stats[corpus_name][ds] = datasets[corpus_name][ds].stats

    # Defining dataloaders
    dataloaders = {}
    for ds in ds_names:
        if 'train' in ds:
            dataloaders[ds] = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([datasets[corpus_name]['train'] for corpus_name in corpora_names]),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
        else:
            for corpus_name in corpora_names:
                if ds not in datasets[corpus_name]:
                    continue
                dataloaders['{0}_{1}'.format(ds, corpus_name)] = torch.utils.data.DataLoader(
                    datasets[corpus_name][ds],
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False
                )
        
    # Defining measures, measure with 0 index is main measure
    measures = [
        MAEMeasure('age_MAE'), # main measure
        UARMeasure('gen_UAR'), 
        MacroF1Measure('gen_MacroF1'),
        PrecisionMeasure('gen_Precision'),
        UARMeasure('mask_UAR'),
    ]
    
    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiment_name = 'w-{4}MultiModalMaskAgender{3}-Scenario{5}-{6}cls-{0}-{1}-{2}'.format(len(c_names['gen']),
                                                          model_cls.__name__.replace('-', '_').replace('/', '_'),
                                                          datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
                                                          {1: 'Early', 2: 'Inter', 3: 'Late'}[int(features_type)], 
                                                          'Augmentation' if augmentation else '',
                                                          t_config.get('SCENARIO', ''),
                                                          mask_num_classes)
    
    # Defining NetTrainer 
    net_trainer = MultiTaskNetTrainer(log_root=logs_root,
                                      experiment_name=experiment_name,
                                      c_names=c_names,
                                      measures=measures,
                                      device=device,                  
                                      final_activations={'gen': nn.Softmax(dim=-1), 'age': nn.Sigmoid(), 'mask': nn.Softmax(dim=-1)},
                                      group_predicts_fn=mask_grouping,
                                      source_code=source_code,
                                      c_names_to_display=c_names_to_display,
                                      use_amp=use_amp)
    
    # Audio (wav2vec2/HuBERT) and visual (ViT) extractors are frozen: their features are pre-extracted
    # by AGenderMultimodalFeaturesDataset. Only the 2 MTCMA blocks + multitask heads are trained.
    model = model_cls(features_type=features_type, checkpoint_path=model_args['checkpoint_path'], mask_num_classes=mask_num_classes)
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print('Trainable parameters: {0} / {1}'.format(sum(p.numel() for p in trainable_params), 
                                                   sum(p.numel() for p in model.parameters())))
    
    # Defining weighted loss: L_total = L_age (CCC) + L_gender (weighted CE) + L_occlusion_type (weighted CE)
    # Class weights are inversely proportional to class frequency on the Train set
    gender_class_sample_count = [sum(x) for x in zip(*[datasets_stats[corpus_name]['train']['counts']['gen'] for corpus_name in corpora_names])]
    mask_class_sample_count = [sum(x) for x in zip(*[datasets_stats[corpus_name]['train']['counts']['mask'] for corpus_name in corpora_names])]
    loss = MaskAgenderLoss(
        mask_weights=compute_inverse_frequency_weights(mask_class_sample_count, device),
        gender_weights=compute_inverse_frequency_weights(gender_class_sample_count, device), 
        gender_alpha=1, 
        age_alpha=1,
        mask_alpha=1,
    )
    
    # Defining optimizer (only trainable parameters)
    optimizer_conf = t_config.get('OPTIMIZER', {'cls': 'AdamW', 'lr': 1e-5})
    optimizer_cls = getattr(torch.optim, optimizer_conf['cls'])
    optimizer = optimizer_cls(trainable_params, lr=optimizer_conf['lr'], betas=(0.9, 0.999), eps=1e-08)

    # Defining scheduler
    scheduler_conf = t_config.get('SCHEDULER', {'T_0': 10, 'T_mult': 2})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=scheduler_conf['T_0'], T_mult=scheduler_conf['T_mult'])

    model, max_perf = net_trainer.run(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders, datasets_stats=datasets_stats, log_epochs=[i for i in range(0, 10)],
                                      early_stopping=t_config.get('EARLY_STOPPING'))

    for phase, perf in max_perf.items():
        if 'train' in phase:
            continue

        print()
        print(phase.capitalize())
        print('Epoch: {}, Max performance:'.format(max_perf[phase][str(measures[0])]['epoch']))
        print([metric for metric in max_perf[phase][str(measures[0])]['performance']])
        print([max_perf[phase][str(measures[0])]['performance'][metric] for metric in max_perf[phase][str(measures[0])]['performance']])
        print()
    

def run_training(scenarios: list[str] = ['A']) -> None:
    """Wrapper for training 

    Args:
        scenarios (list[str], optional): Scenarios to run: 'A' (BRAVE-MASKS, 6 classes) and/or 
                                         'B' (BRAVE-MASKS + DIVINA, 10 classes). Defaults to ['A', 'B'].
    """
    d_config = ensure_configs(deepcopy(dconf))
    
    models = [
        # {'cls': MaskAgenderAVModelV1, 'checkpoint_path': 'w-AugmentationMultiModalFinalInter-2-AVModelV3-2024.10.18-08.34.16/models/epoch_1.pth'},
        # {'cls': MaskAgenderAVModelV2, 'checkpoint_path': 'w-AugmentationMultiModalFinalInter-2-AVModelV3-2024.10.18-08.34.16/models/epoch_1.pth'},
        # MaskAgenderAVModelV3: trainable MTCMA blocks on frozen (pre-extracted) audio/visual features.
        # NOTE: upstream referenced a non-existent MaskAgenderAVModelV4.
        {'cls': MaskAgenderAVModelV3, 'checkpoint_path': None},
    ]

    features_types = [FeaturesType.INTERMEDIATE]
    
    for scenario in scenarios:
        for features_type in features_types:
            for idx, model_info in enumerate(models):
                corpora_names = SCENARIOS[scenario]['corpora']
                new_d_config = {corpus_name: d_config[corpus_name] for corpus_name in corpora_names}
                
                t_config = deepcopy(tconf)
                t_config['SCENARIO'] = scenario
                t_config['MASK_NUM_CLASSES'] = SCENARIOS[scenario]['mask_num_classes']
                t_config['FEATURE_EXTRACTOR']['FEATURES_TYPE'] = features_type

                t_config['FEATURE_EXTRACTOR']['FEATURES_FILE_NAME'] = {1: 'EARLYF_SAMPLES', 2: 'INTERF_SAMPLES', 3: 'LATEF_SAMPLES'}[int(features_type)]
                    
                t_config['MODEL']['cls'] = model_info['cls']
                t_config['MODEL']['args'] = {'checkpoint_path': model_info['checkpoint_path']}
                # t_config['MODEL']['args'] = {'checkpoint_path': os.path.join('/media/maxim/WesternDigital/aGender2024/MultimodalNoMask', model_info['checkpoint_path'])}
                t_config['AUGMENTATION'] = True
                t_config['NUM_EPOCHS'] = t_config.get('NUM_EPOCHS', 500)
                t_config['BATCH_SIZE'] = 16
                t_config['USE_AMP'] = t_config.get('USE_AMP', True)
                
                main(d_config=new_d_config, t_config=t_config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pelatihan Model Multimodal ORAGEN (Skenario A & B)")
    parser.add_argument('--scenario', default=None, choices=['A', 'B'],
                        help="Skenario: 'A' (Baseline BRAVE-MASKS 6 kelas) atau 'B' (BRAVE-MASKS + DIVINA 10 kelas)")
    parser.add_argument('--epochs', type=int, default=None, help="Override jumlah epoch")
    parser.add_argument('--patience', type=int, default=None, help="Patience Early Stopping")
    args = parser.parse_args()

    active_scenario = args.scenario if args.scenario else tconf.get('SCENARIO', 'A')
    if args.epochs:
        tconf['NUM_EPOCHS'] = args.epochs
    if args.patience:
        if 'EARLY_STOPPING' not in tconf or not isinstance(tconf['EARLY_STOPPING'], dict):
            tconf['EARLY_STOPPING'] = {'min_delta': 0.05}
        tconf['EARLY_STOPPING']['patience'] = args.patience

    print(f"=== Menjalankan Skenario: {active_scenario} ===")
    run_training(scenarios=[active_scenario])
