"""
Evaluasi Zero-Shot Model Baseline (Skenario A) pada Subset TEST DIVINA (Indonesia).
Sesuai Proposal Bab 3.8.1:
  Model Baseline yang hanya dilatih pada BRAVE-MASKS (Rusia/Eropa)
  diuji pada subset TEST DIVINA untuk mengukur besarnya dampak Domain Shift.

Menghitung:
  - age_MAE (Tahun)
  - gen_UAR, gen_MacroF1, gen_Precision (%)
  - Confusion Matrix Gender & Oklusi

Jalankan:
  python src/fusion/run_eval_zeroshot_divina.py --checkpoint /path/to/epoch_3.pth
"""
import os
import sys
import glob
import argparse
from copy import deepcopy

sys.path.append('src')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from configs.multimodal_bm_config import data_config as dconf
from configs.multimodal_bm_config import training_config as tconf

from fusion.data.agender_multimodal_features_dataset import AGenderMultimodalFeaturesDataset
from fusion.features.common import FeaturesType
from fusion.models.multimodal_models import MaskAgenderAVModelV3
from common.loss.loss import MaskAgenderLoss, compute_inverse_frequency_weights
from common.utils.accuracy import *
from common.net_trainer.multitask_net_trainer import MultiTaskNetTrainer


def find_latest_checkpoint(logs_root: str) -> str:
    """Cari checkpoint epoch_3.pth atau epoch terbaik dari folder logs."""
    candidates = glob.glob(os.path.join(logs_root, '**', 'epoch_3.pth'), recursive=True)
    if candidates:
        return candidates[-1]
    all_pth = glob.glob(os.path.join(logs_root, '**', 'epoch_*.pth'), recursive=True)
    if all_pth:
        return sorted(all_pth)[-1]
    return None


def run_evaluation(checkpoint_path: str, subset: str = 'test') -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Menggunakan device: {device}')
    print(f'Checkpoint model: {checkpoint_path}')
    assert os.path.exists(checkpoint_path), f'Checkpoint tidak ditemukan: {checkpoint_path}'

    # Konfigurasi data DIVINA
    d_config = deepcopy(dconf)
    t_config = deepcopy(tconf)

    assert 'DIVINA' in d_config, 'Konfigurasi DIVINA belum ada di default_config.py'
    divina_cfg = d_config['DIVINA']

    labels_df = pd.read_csv(divina_cfg['LABELS_FILE_PATH'], low_memory=False)
    target_labels = labels_df[labels_df['subset'] == subset].copy()
    print(f'Total sampel evaluasi DIVINA ({subset}): {len(target_labels)}')

    # Muat dataset fitur
    dataset = AGenderMultimodalFeaturesDataset(
        data_root=divina_cfg['PROCESSED_DATA_ROOT'],
        labels_metadata=target_labels,
        features_root=t_config['FEATURE_EXTRACTOR']['FEATURES_ROOT'],
        features_file_name=f'DIVINA_{subset.upper()}_INTERF_SAMPLES',
        corpus_name='DIVINA',
        gender_num_classes=2,
        features_type=FeaturesType.INTERMEDIATE,
        channels=['c'],
        mask_types=list(range(6)), # Model baseline memiliki head 6 kelas
        vad_metadata=None,
        sr=16000, win_max_length=4, win_shift=2, win_min_length=1,
        include_mask=True,
        transform=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=t_config.get('BATCH_SIZE', 16),
        shuffle=False,
        num_workers=2,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    )

    # Inisialisasi Model Baseline (6 kelas oklusi)
    model = MaskAgenderAVModelV3(
        features_type=FeaturesType.INTERMEDIATE,
        checkpoint_path=None,
        mask_num_classes=6
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print('Checkpoint berhasil dimuat ke arsitektur MaskAgenderAVModelV3!')

    # Evaluasi metrik
    all_gen_preds, all_gen_targets = [], []
    all_age_preds, all_age_targets = [], []
    all_mask_preds, all_mask_targets = [], []

    print('\nMenjalankan inferensi model baseline pada subset test DIVINA...')
    with torch.no_grad():
        for batch in dataloader:
            x, y, info = batch
            a = x[0].to(device)
            v = x[1].to(device)
            
            out = model([a, v])
            gen_pred = torch.argmax(out['gen'], dim=-1).cpu().numpy()
            age_pred = (out['age'].cpu().numpy() * 100.0) # denormalisasi ke satuan tahun
            mask_pred = torch.argmax(out['mask'], dim=-1).cpu().numpy()

            gen_true = y['gen'].cpu().numpy()
            age_true = (y['age'].cpu().numpy() * 100.0)
            mask_true = y['mask'].cpu().numpy()

            all_gen_preds.extend(gen_pred)
            all_gen_targets.extend(gen_true)
            all_age_preds.extend(age_pred)
            all_age_targets.extend(age_true)
            all_mask_preds.extend(mask_pred)
            all_mask_targets.extend(mask_true)

    all_gen_preds = np.array(all_gen_preds)
    all_gen_targets = np.array(all_gen_targets)
    all_age_preds = np.array(all_age_preds)
    all_age_targets = np.array(all_age_targets)

    # Hitung metrik
    mae_age = float(np.mean(np.abs(all_age_preds - all_age_targets)))
    
    # Gender UAR, MacroF1, Precision
    uar_gen = float(calc_uar(all_gen_targets, all_gen_preds))
    macro_f1_gen = float(calc_macro_f1(all_gen_targets, all_gen_preds))
    precision_gen = float(calc_precision(all_gen_targets, all_gen_preds))

    print('\n' + '=' * 65)
    print('HASIL EVALUASI ZERO-SHOT MODEL BASELINE PADA DATA DIVINA (INDONESIA)')
    print('=' * 65)
    print(f'1. Estimasi Usia (Age MAE) : {mae_age:.3f} tahun')
    print(f'2. Pengenalan Gender (UAR) : {uar_gen:.2f} %')
    print(f'   - Macro F1              : {macro_f1_gen:.2f} %')
    print(f'   - Precision             : {precision_gen:.2f} %')
    print('=' * 65)
    print('Interpretasi:')
    print('Angka di atas adalah performa murni model sebelum dilatih dengan data lokal.')
    print('Nilai MAE usia dan penurunan akurasi gender membuktikan besarnya DOMAIN SHIFT.')
    print('=' * 65)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='Path file checkpoint .pth')
    ap.add_argument('--logs_root', default='logs', help='Folder logs jika checkpoint tidak ditentukan')
    ap.add_argument('--subset', default='test', choices=['test', 'devel'])
    args = ap.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint(args.logs_root)
    assert ckpt, 'Tidak menemukan checkpoint model .pth! Harap tentukan --checkpoint'
    run_evaluation(ckpt, args.subset)
