"""
Evaluasi Lengkap Model ORAGEN (Skenario A Zero-Shot & Skenario B Diperluas) pada Dataset DIVINA.
Sesuai Proposal Tugas Akhir Bab 3.8.1, 3.8.2, dan 3.8.3:
  1. Evaluasi Skenario A (Baseline 6 kelas, dilatih hanya pada BRAVE-MASKS) -> diuji pada Test DIVINA & Test BRAVE-MASKS.
  2. Evaluasi Skenario B (Model Diperluas 10 kelas, dilatih pada BRAVE-MASKS + DIVINA) -> diuji pada Test DIVINA & Test BRAVE-MASKS.
  3. Analisis Tambahan 1 (Bab 3.8.3): Performa per jenis oklusi lokal (NO, HJ, HM, KA, TP) pada Test DIVINA.
  4. Analisis Tambahan 2 (Bab 3.8.3): Kontribusi modalitas (Audio-Visual Fusion vs Audio-Only vs Video-Only) pada Test DIVINA.
  5. Perhitungan Besaran Domain Shift yang Teratasi (Selisih MAE Usia & UAR Gender/Oklusi antara Skenario A dan B).

Jalankan:
  python src/fusion/run_eval_zeroshot_divina.py --scenario A
  python src/fusion/run_eval_zeroshot_divina.py --scenario B
  python src/fusion/run_eval_zeroshot_divina.py --compare
"""
import os
import sys
import glob
import json
import pickle
import argparse
from copy import deepcopy

sys.path.append('src')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from configs.multimodal_bm_config import data_config as dconf
from configs.multimodal_bm_config import training_config as tconf

from fusion.data.agender_multimodal_features_dataset import AGenderMultimodalFeaturesDataset
from fusion.features.common import FeaturesType
from fusion.models.multimodal_models import MaskAgenderAVModelV3
from common.data.grouping import mask_grouping
from common.utils.accuracy import MAEMeasure, UARMeasure, MacroF1Measure, PrecisionMeasure
from common.visualization.visualize import conf_matrix, plot_conf_matrix


OCC_CODE_TO_NAME = {
    'NO': 'Tanpa Oklusi (No mask)',
    'HJ': 'Hijab',
    'HM': 'Helm Motor',
    'KA': 'Kacamata',
    'TP': 'Topi',
}


def find_best_checkpoint(logs_root: str, scenario: str = 'A') -> tuple[str, int]:
    """Cari checkpoint (.pth) pada epoch terbaik (MAE evaluasi terendah) untuk skenario A atau B."""
    tag = f'Scenario{scenario}'
    exp_dirs = [
        d for d in glob.glob(os.path.join(logs_root, '*'))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'logs', 'stats.csv'))
    ]
    scen_dirs = [d for d in exp_dirs if tag in os.path.basename(d)]
    if not scen_dirs and scenario == 'A':
        scen_dirs = [d for d in exp_dirs if 'ScenarioB' not in os.path.basename(d)]
    if not scen_dirs:
        return None, None

    scen_dirs = sorted(scen_dirs, key=os.path.getmtime)
    for exp_dir in reversed(scen_dirs):
        stats_fp = os.path.join(exp_dir, 'logs', 'stats.csv')
        df = pd.read_csv(stats_fp, sep=';')
        # Prioritaskan devel_age_MAE jika ada, kalau tidak test_age_MAE
        devel_cols = [c for c in df.columns if c.endswith('_devel_age_MAE')]
        test_cols = [c for c in df.columns if c.endswith('_test_age_MAE')]
        eval_cols = devel_cols if devel_cols else test_cols
        if not eval_cols:
            continue

        # Urutkan epoch dari MAE terbaik, ambil epoch pertama yang file .pth-nya tersedia
        df_sorted = df.loc[df[eval_cols].mean(axis=1).sort_values().index]
        for ep in df_sorted['epoch'].astype(int):
            pths = glob.glob(os.path.join(exp_dir, 'models', '**', f'epoch_{ep}.pth'), recursive=True)
            if pths:
                return pths[0], int(ep)

        all_pths = sorted(glob.glob(os.path.join(exp_dir, 'models', '**', 'epoch_*.pth'), recursive=True))
        if all_pths:
            ep = int(os.path.basename(all_pths[0]).replace('epoch_', '').replace('.pth', ''))
            return all_pths[0], ep

    return None, None


def load_corpus_dataset(corpus_name: str, subset: str, d_config: dict, t_config: dict) -> AGenderMultimodalFeaturesDataset:
    cfg = d_config[corpus_name]
    labels_df = pd.read_csv(cfg['LABELS_FILE_PATH'], low_memory=False)
    with open(cfg['VAD_FILE_PATH'], 'rb') as f:
        vad_meta = pickle.load(f)

    sub_df = labels_df[labels_df['subset'] == subset].copy()
    feat_cfg = t_config['FEATURE_EXTRACTOR']
    # Gunakan nama cache standar agar 100% kompatibel & tidak ekstrak dua kali
    feat_fn = f"{corpus_name}_{subset.upper()}_INTERF_SAMPLES"
    return AGenderMultimodalFeaturesDataset(
        data_root=cfg['PROCESSED_DATA_ROOT'],
        labels_metadata=sub_df,
        features_root=feat_cfg['FEATURES_ROOT'],
        features_file_name=feat_fn,
        corpus_name=corpus_name,
        gender_num_classes=2,
        features_type=FeaturesType.INTERMEDIATE,
        channels=['c'],
        mask_types=list(range(10)),  # Muat semua kelas oklusi (0..9)
        vad_metadata=vad_meta,
        sr=feat_cfg.get('SR', 16000),
        win_max_length=feat_cfg.get('WIN_MAX_LENGTH', 4),
        win_shift=feat_cfg.get('WIN_SHIFT', 2),
        win_min_length=feat_cfg.get('WIN_MIN_LENGTH', 1),
        include_mask=True,
        transform=None,
    )


def infer_grouped(model: nn.Module, dataset: AGenderMultimodalFeaturesDataset,
                  device: torch.device, batch_size: int = 16,
                  modality: str = 'both') -> tuple[dict, dict, list[str]]:
    """Inferensi dengan aktivasi final + file-level grouping (persis seperti MultiTaskNetTrainer)."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    final_act = {
        'gen': nn.Softmax(dim=-1),
        'age': nn.Sigmoid(),
        'mask': nn.Softmax(dim=-1),
    }
    targets = {'gen': [], 'age': [], 'mask': []}
    predicts = {'gen': [], 'age': [], 'mask': []}
    sample_info = []

    model.eval()
    with torch.no_grad():
        for inps, labs, s_info in tqdm(dataloader, desc=f'Inferensi ({dataset.corpus_name}-{modality})', leave=False):
            a, v = inps[0].to(device), inps[1].to(device)
            if modality == 'audio':
                v = torch.zeros_like(v)
            elif modality == 'video':
                a = torch.zeros_like(a)

            out = model([a, v])
            preds = {t: final_act[t](out[t].float()) for t in ('gen', 'age', 'mask')}
            for t in ('gen', 'age', 'mask'):
                targets[t].extend(labs[t].cpu().numpy())
                predicts[t].extend(preds[t].cpu().numpy())
            sample_info.extend(s_info)

    grouped = mask_grouping(targets, predicts, sample_info)
    return grouped[dataset.corpus_name]


def compute_metrics(t_dict: dict, p_dict: dict, eval_mask: bool = True) -> dict:
    mae_fn = MAEMeasure('age_MAE')
    uar_fn = UARMeasure('UAR')
    f1_fn = MacroF1Measure('MacroF1')
    prec_fn = PrecisionMeasure('Precision')

    res = {
        'age_MAE': round(float(mae_fn(t_dict['age'], p_dict['age'])), 3),
        'gen_UAR': round(float(uar_fn(t_dict['gen'], p_dict['gen'])), 2),
        'gen_MacroF1': round(float(f1_fn(t_dict['gen'], p_dict['gen'])), 2),
        'gen_Precision': round(float(prec_fn(t_dict['gen'], p_dict['gen'])), 2),
        'mask_UAR': round(float(uar_fn(t_dict['mask'], p_dict['mask'])), 2) if eval_mask else None,
        'mask_MacroF1': round(float(f1_fn(t_dict['mask'], p_dict['mask'])), 2) if eval_mask else None,
    }
    return res


def evaluate_scenario(scenario: str, checkpoint_path: str = None, subset: str = 'test',
                      fig_dir: str = None) -> dict:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    d_config = deepcopy(dconf)
    t_config = deepcopy(tconf)
    logs_root = t_config['LOGS_ROOT']
    if fig_dir is None:
        fig_dir = os.path.join(os.path.dirname(logs_root), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    best_epoch = None
    if not checkpoint_path:
        checkpoint_path, best_epoch = find_best_checkpoint(logs_root, scenario=scenario)
    assert checkpoint_path and os.path.exists(checkpoint_path), (
        f"Checkpoint untuk Skenario {scenario} belum ditemukan di {logs_root}."
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # Deteksi otomatis jumlah kelas head oklusi dari bobot checkpoint (6 untuk Skenario A, 10 untuk Skenario B)
    mask_out_key = [k for k in state_dict.keys() if 'mask' in k and k.endswith('weight')][-1]
    mask_num_classes = int(state_dict[mask_out_key].shape[0])

    model = MaskAgenderAVModelV3(
        features_type=FeaturesType.INTERMEDIATE,
        checkpoint_path=None,
        mask_num_classes=mask_num_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print('\n' + '=' * 75)
    print(f'EVALUASI SKENARIO {scenario} ({mask_num_classes} KELAS OKLUSI)')
    print(f'Checkpoint : {checkpoint_path} (Epoch terbaik: {best_epoch or ckpt.get("epoch", "?")})')
    print(f'Device     : {device}')
    print('=' * 75)

    results = {'scenario': scenario, 'checkpoint': checkpoint_path, 'epoch': best_epoch, 'mask_num_classes': mask_num_classes}

    # 0) Ambil hasil evaluasi pada Test BRAVE-MASKS dari stats.csv pada epoch terbaik
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    stats_fp = os.path.join(exp_dir, 'logs', 'stats.csv')
    if os.path.exists(stats_fp):
        df_st = pd.read_csv(stats_fp, sep=';')
        ep_num = best_epoch or ckpt.get('epoch')
        row_bm = df_st[df_st['epoch'] == ep_num]
        if len(row_bm):
            r0 = row_bm.iloc[0]
            bm_metrics = {
                'age_MAE': round(float(r0.get('BRAVEMASKS_test_age_MAE', 0)), 3),
                'gen_UAR': round(float(r0.get('BRAVEMASKS_test_gen_UAR', 0)), 2),
                'gen_MacroF1': round(float(r0.get('BRAVEMASKS_test_gen_MacroF1', 0)), 2),
                'gen_Precision': round(float(r0.get('BRAVEMASKS_test_gen_Precision', 0)), 2),
                'mask_UAR': round(float(r0.get('BRAVEMASKS_test_mask_UAR', 0)), 2),
            }
            results['BRAVEMASKS_test'] = bm_metrics
            print(f'\n[0] Hasil Evaluasi pada Subset TEST BRAVE-MASKS (Epoch {ep_num}):')
            print(f"    - Estimasi Usia (age_MAE)     : {bm_metrics['age_MAE']:.3f} tahun")
            print(f"    - Pengenalan Gender (gen_UAR) : {bm_metrics['gen_UAR']:.2f} % (Macro F1: {bm_metrics['gen_MacroF1']:.2f} %, Precision: {bm_metrics['gen_Precision']:.2f} %)")
            print(f"    - Pengenalan Oklusi (mask_UAR): {bm_metrics['mask_UAR']:.2f} %")

    # 1) Evaluasi pada Test DIVINA
    assert 'DIVINA' in d_config, "Konfigurasi 'DIVINA' belum ada di default_config.py"
    ds_divina = load_corpus_dataset('DIVINA', subset, d_config, t_config)
    t_div, p_div, fns_div = infer_grouped(model, ds_divina, device, t_config.get('BATCH_SIZE', 16), modality='both')

    # Untuk Skenario A (6 kelas), model hanya mengenal kelas 0 (No mask) dari DIVINA; kelas 6..9 belum pernah dilatih
    m_div = compute_metrics(t_div, p_div, eval_mask=(mask_num_classes == 10))
    # Hitung akurasi No-Mask pada Skenario A untuk sampel polos (kelas 0)
    no_mask_idx = [i for i, y in enumerate(t_div['mask']) if int(y) == 0]
    if no_mask_idx:
        nm_acc = round(100.0 * float(np.mean([int(p_div['mask'][i]) == 0 for i in no_mask_idx])), 2)
        m_div['no_mask_recall_only'] = nm_acc

    results['DIVINA_both'] = m_div

    print(f'\n[1] Hasil Utama pada Subset {subset.upper()} DIVINA ({len(fns_div)} klip):')
    print(f"    - Estimasi Usia (age_MAE)     : {m_div['age_MAE']:.3f} tahun")
    print(f"    - Pengenalan Gender (gen_UAR) : {m_div['gen_UAR']:.2f} %")
    print(f"    - Gender Macro F1             : {m_div['gen_MacroF1']:.2f} %")
    print(f"    - Gender Precision            : {m_div['gen_Precision']:.2f} %")
    if m_div['mask_UAR'] is not None:
        print(f"    - Pengenalan Oklusi (mask_UAR): {m_div['mask_UAR']:.2f} % (Macro F1: {m_div['mask_MacroF1']:.2f} %)")
    else:
        print(f"    - Pengenalan Oklusi           : N/A pada 4 oklusi lokal (Model Baseline 6 kelas belum dilatih Hijab/Helm/Kacamata/Topi; Recall pada kelas Tanpa Oklusi/Polos = {m_div.get('no_mask_recall_only', 0):.2f} %)")

    # Simpan Confusion Matrix Gender & Oklusi pada Test DIVINA
    cm_gen = conf_matrix(t_div['gen'], p_div['gen'], ['Female', 'Male'])
    plot_conf_matrix(
        cm_gen, labels=['Female', 'Male'], xticks_rotation=45,
        title=f"Scenario {scenario} - Gender (DIVINA {subset}). age_MAE={m_div['age_MAE']:.2f}y, UAR={m_div['gen_UAR']:.2f}%",
        save_path=os.path.join(fig_dir, f'Scenario{scenario}_DIVINA_{subset}_cm_gender.png')
    )
    if mask_num_classes == 10:
        c_mask_all = [cn.capitalize() for cn in d_config['DIVINA']['C_NAMES']['mask'][:10]]
        cm_mask = conf_matrix(t_div['mask'], p_div['mask'], c_mask_all)
        plot_conf_matrix(
            cm_mask, labels=c_mask_all, xticks_rotation=45, figsize=(10, 8),
            title=f"Scenario {scenario} - Occlusion 10-Class (DIVINA {subset}). UAR={m_div['mask_UAR']:.2f}%",
            save_path=os.path.join(fig_dir, f'Scenario{scenario}_DIVINA_{subset}_cm_mask10.png')
        )

    # 2) Analisis Tambahan 1 (Proposal Bab 3.8.3): Breakdown per Jenis Oklusi (NO, HJ, HM, KA, TP)
    labels_df = pd.read_csv(d_config['DIVINA']['LABELS_FILE_PATH'], low_memory=False)
    fn_to_occ = labels_df.drop_duplicates('audio_file_path').set_index('audio_file_path')['occlusion_code'].to_dict()

    occ_rows = []
    mae_fn = MAEMeasure('age_MAE')
    uar_fn = UARMeasure('UAR')
    for occ_code in ['NO', 'HJ', 'HM', 'KA', 'TP']:
        idxs = [i for i, fn in enumerate(fns_div) if fn_to_occ.get(fn) == occ_code]
        if not idxs:
            continue
        t_age_sub = [t_div['age'][i] for i in idxs]
        p_age_sub = [p_div['age'][i] for i in idxs]
        t_gen_sub = [t_div['gen'][i] for i in idxs]
        p_gen_sub = [p_div['gen'][i] for i in idxs]
        t_mask_sub = [t_div['mask'][i] for i in idxs]
        p_mask_sub = [p_div['mask'][i] for i in idxs]

        gen_acc = round(100.0 * float(np.mean(np.array(t_gen_sub) == np.array(p_gen_sub))), 2)
        gen_uar_sub = round(float(uar_fn(t_gen_sub, p_gen_sub)), 2)
        occ_acc = round(100.0 * float(np.mean(np.array(t_mask_sub) == np.array(p_mask_sub))), 2) if mask_num_classes == 10 else (
            round(100.0 * float(np.mean(np.array(p_mask_sub) == 0)), 2) if occ_code == 'NO' else None
        )
        occ_rows.append({
            'Kode': occ_code,
            'Jenis Oklusi': OCC_CODE_TO_NAME[occ_code],
            'Jumlah Klip': len(idxs),
            'Age MAE (tahun)': round(float(mae_fn(t_age_sub, p_age_sub)), 3),
            'Gender UAR (%)': gen_uar_sub,
            'Gender Acc (%)': gen_acc,
            'Oklusi Recall (%)': occ_acc,
        })
    df_occ = pd.DataFrame(occ_rows)
    df_occ.to_csv(os.path.join(fig_dir, f'Scenario{scenario}_DIVINA_{subset}_per_occlusion.csv'), index=False)
    results['per_occlusion'] = occ_rows

    print(f'\n[2] Analisis Per Jenis Oklusi pada {subset.upper()} DIVINA (Proposal Bab 3.8.3 Poin 1):')
    print(df_occ.to_string(index=False))

    # 3) Analisis Tambahan 2 (Proposal Bab 3.8.3): Kontribusi Modalitas (Audio-Visual vs Audio-Only vs Video-Only)
    mod_rows = []
    for mod_key, mod_label in [('both', 'Audio-Visual (MTCMA Fusion)'), ('audio', 'Audio-Only (Wav2Vec2.0)'), ('video', 'Visual-Only (ViT)')]:
        if mod_key == 'both':
            m_mod = m_div
        else:
            t_m, p_m, _ = infer_grouped(model, ds_divina, device, t_config.get('BATCH_SIZE', 16), modality=mod_key)
            m_mod = compute_metrics(t_m, p_m, eval_mask=(mask_num_classes == 10))
        mod_rows.append({
            'Modalitas': mod_label,
            'Age MAE (tahun)': m_mod['age_MAE'],
            'Gender UAR (%)': m_mod['gen_UAR'],
            'Gender MacroF1 (%)': m_mod['gen_MacroF1'],
            'Oklusi UAR (%)': m_mod['mask_UAR'],
        })
    df_mod = pd.DataFrame(mod_rows)
    df_mod.to_csv(os.path.join(fig_dir, f'Scenario{scenario}_DIVINA_{subset}_per_modality.csv'), index=False)
    results['per_modality'] = mod_rows

    print(f'\n[3] Analisis Kontribusi Modalitas pada {subset.upper()} DIVINA (Proposal Bab 3.8.3 Poin 2):')
    print(df_mod.to_string(index=False))

    out_json = os.path.join(fig_dir, f'Scenario{scenario}_DIVINA_{subset}_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nRingkasan tabel & Confusion Matrix disimpan di: {fig_dir}')
    return results


def compare_scenarios(subset: str = 'test', fig_dir: str = None) -> None:
    """Bandingkan Skenario A vs Skenario B pada Test DIVINA untuk mengukur Domain Shift yang teratasi."""
    t_config = deepcopy(tconf)
    logs_root = t_config['LOGS_ROOT']
    if fig_dir is None:
        fig_dir = os.path.join(os.path.dirname(logs_root), 'figures')

    json_a = os.path.join(fig_dir, f'ScenarioA_DIVINA_{subset}_summary.json')
    json_b = os.path.join(fig_dir, f'ScenarioB_DIVINA_{subset}_summary.json')

    res_a = json.load(open(json_a, encoding='utf-8')) if os.path.exists(json_a) else evaluate_scenario('A', subset=subset, fig_dir=fig_dir)
    res_b = json.load(open(json_b, encoding='utf-8')) if os.path.exists(json_b) else evaluate_scenario('B', subset=subset, fig_dir=fig_dir)

    ma, mb = res_a['DIVINA_both'], res_b['DIVINA_both']
    comp_rows = [
        {
            'Metrik Evaluasi (Test DIVINA)': 'Estimasi Usia - MAE (tahun) [Semakin kecil semakin baik]',
            'Skenario A (Baseline Tanpa DIVINA)': ma['age_MAE'],
            'Skenario B (Diperluas + DIVINA)': mb['age_MAE'],
            'Domain Shift Teratasi (Delta)': f"{mb['age_MAE'] - ma['age_MAE']:+.3f} tahun",
        },
        {
            'Metrik Evaluasi (Test DIVINA)': 'Pengenalan Gender - UAR (%) [Semakin besar semakin baik]',
            'Skenario A (Baseline Tanpa DIVINA)': ma['gen_UAR'],
            'Skenario B (Diperluas + DIVINA)': mb['gen_UAR'],
            'Domain Shift Teratasi (Delta)': f"{mb['gen_UAR'] - ma['gen_UAR']:+.2f} %",
        },
        {
            'Metrik Evaluasi (Test DIVINA)': 'Pengenalan Gender - Macro F1 (%)',
            'Skenario A (Baseline Tanpa DIVINA)': ma['gen_MacroF1'],
            'Skenario B (Diperluas + DIVINA)': mb['gen_MacroF1'],
            'Domain Shift Teratasi (Delta)': f"{mb['gen_MacroF1'] - ma['gen_MacroF1']:+.2f} %",
        },
        {
            'Metrik Evaluasi (Test DIVINA)': 'Pengenalan Oklusi (10 Kelas) - UAR (%)',
            'Skenario A (Baseline Tanpa DIVINA)': 'N/A (Hanya 6 kelas masker)',
            'Skenario B (Diperluas + DIVINA)': f"{mb['mask_UAR']} %",
            'Domain Shift Teratasi (Delta)': f"+{mb['mask_UAR']} % (Kemampuan baru 4 oklusi lokal)",
        },
    ]
    df_comp = pd.DataFrame(comp_rows)
    comp_csv = os.path.join(fig_dir, f'Comparison_ScenarioA_vs_ScenarioB_DIVINA_{subset}.csv')
    df_comp.to_csv(comp_csv, index=False)

    print('\n' + '=' * 85)
    print('TABEL UTAMA BAB 4: PERBANDINGAN SKENARIO A VS SKENARIO B PADA TEST DIVINA (INDONESIA)')
    print('=' * 85)
    print(df_comp.to_string(index=False))
    print('=' * 85)
    print(f'Tabel komparasi disimpan ke: {comp_csv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='A', choices=['A', 'B'], help='Evaluasi Skenario A atau B')
    ap.add_argument('--checkpoint', default=None, help='Path file checkpoint .pth (opsional, otomatis dicari jika kosong)')
    ap.add_argument('--logs_root', default=None, help='Override folder logs_root')
    ap.add_argument('--subset', default='test', choices=['test', 'devel'])
    ap.add_argument('--compare', action='store_true', help='Bandingkan Skenario A vs Skenario B sekaligus')
    args = ap.parse_args()

    if args.logs_root:
        tconf['LOGS_ROOT'] = args.logs_root

    if args.compare:
        compare_scenarios(subset=args.subset)
    else:
        evaluate_scenario(args.scenario, checkpoint_path=args.checkpoint, subset=args.subset)
