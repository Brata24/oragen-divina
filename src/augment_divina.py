"""
Augmentasi Dataset DIVINA - Skenario Opsi B (Hanya pada Subset TRAIN).

Sesuai Tabel 3.0.9 & 3.0.10 Proposal Tugas Akhir:
Setiap sampel pada subset TRAIN menghasilkan 7 variasi (1 Original + 6 Augmentasi):
  - orig : Audio Asli                      | Video Asli
  - aug1 : AWGN (SNR 30 dB)                | Video Asli
  - aug2 : Gain +3 dB                      | Video Asli
  - aug3 : Audio Compression (64 kbps)     | Video Asli
  - aug4 : Audio Asli                      | Brightness +20%
  - aug5 : Audio Asli                      | Gaussian Blur (3x3)
  - aug6 : Audio Asli                      | Video Compression (1 Mbps / DCT Quantization)

Subset DEVEL dan TEST dibiarkan 100% Asli (Original) agar evaluasi valid dan bebas bias sintetis.

Jalankan:
  python src/augment_divina.py
"""
import os
import re
import glob
import shutil
import pickle
import hashlib
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
import imageio_ffmpeg
from tqdm import tqdm

SR = 16000
GAIN_3DB = float(10.0 ** (3.0 / 20.0))  # +3 dB = ~1.4125375x amplitudo
SNR_DB = 30.0                           # AWGN SNR 30 dB
BRIGHTNESS_FACTOR = 1.20                # Brightness +20%
BLUR_KERNEL = (3, 3)                    # Gaussian Blur 3x3
# Kuantisasi DCT setara bit-per-pixel video 1080p @ 1 Mbps pada crop wajah 224x224
VIDEO_COMP_JPEG_QUALITY = 30


def stable_seed(text: str) -> int:
    """Seed deterministik per klip agar hasil AWGN selalu konsisten & reproducible."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)


def aug_awgn(wav: torch.Tensor, seed_key: str, snr_db: float = SNR_DB) -> torch.Tensor:
    """Aug-1: Additive White Gaussian Noise (AWGN) dengan SNR 30 dB."""
    g = torch.Generator()
    g.manual_seed(stable_seed(seed_key))
    sig_power = torch.mean(wav ** 2).clamp(min=1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(wav.shape, generator=g, dtype=wav.dtype) * torch.sqrt(noise_power)
    return torch.clamp(wav + noise, -1.0, 1.0)


def aug_gain(wav: torch.Tensor, factor: float = GAIN_3DB) -> torch.Tensor:
    """Aug-2: Gain Perturbation +3 dB."""
    return torch.clamp(wav * factor, -1.0, 1.0)


def aug_audio_compression_64k(src_wav_fp: str, dst_wav_fp: str, ffmpeg_exe: str) -> None:
    """Aug-3: Audio Compression 64 kbps (encode MP3 64 kbps -> decode kembali ke WAV 16 kHz mono PCM16)."""
    tmp_mp3 = dst_wav_fp + '.tmp.mp3'
    try:
        cmd_enc = [
            ffmpeg_exe, '-y', '-loglevel', 'error',
            '-i', src_wav_fp, '-vn', '-ar', str(SR), '-ac', '1',
            '-codec:a', 'libmp3lame', '-b:a', '64k', tmp_mp3
        ]
        subprocess.run(cmd_enc, check=True)
        cmd_dec = [
            ffmpeg_exe, '-y', '-loglevel', 'error',
            '-i', tmp_mp3, '-vn', '-ar', str(SR), '-ac', '1',
            '-c:a', 'pcm_s16le', dst_wav_fp
        ]
        subprocess.run(cmd_dec, check=True)
    finally:
        if os.path.exists(tmp_mp3):
            try:
                os.remove(tmp_mp3)
            except OSError:
                pass


def aug_brightness(img: np.ndarray, factor: float = BRIGHTNESS_FACTOR) -> np.ndarray:
    """Aug-4: Brightness Adjustment +20%."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def aug_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Aug-5: Gaussian Blur kernel 3x3."""
    return cv2.GaussianBlur(img, BLUR_KERNEL, 0)


def aug_video_compression(img: np.ndarray, dst_fp: str, quality: int = VIDEO_COMP_JPEG_QUALITY) -> None:
    """Aug-6: Video Compression (simulasi artefak kompresi 1 Mbps pada crop 224x224)."""
    cv2.imwrite(dst_fp, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def process_one_clip(clip_rel_audio: str, img_rel_list: list[str], out_root: str, ffmpeg_exe: str) -> None:
    """Membuat file fisik untuk 6 variasi augmentasi dari 1 klip train."""
    src_wav_fp = os.path.join(out_root, clip_rel_audio)
    stem_rel = clip_rel_audio[:-4]  # tanpa '.wav', mis: audio/train/M01_helm_001

    wav1_fp = os.path.join(out_root, f'{stem_rel}__aug1.wav')
    wav2_fp = os.path.join(out_root, f'{stem_rel}__aug2.wav')
    wav3_fp = os.path.join(out_root, f'{stem_rel}__aug3.wav')

    if not (os.path.exists(wav1_fp) and os.path.exists(wav2_fp)):
        wav, sr = torchaudio.load(src_wav_fp)
        wav = wav.mean(0)
        if not os.path.exists(wav1_fp):
            w1 = aug_awgn(wav, clip_rel_audio).unsqueeze(0)
            torchaudio.save(wav1_fp, w1, SR, encoding='PCM_S', bits_per_sample=16)
        if not os.path.exists(wav2_fp):
            w2 = aug_gain(wav).unsqueeze(0)
            torchaudio.save(wav2_fp, w2, SR, encoding='PCM_S', bits_per_sample=16)

    if not os.path.exists(wav3_fp):
        aug_audio_compression_64k(src_wav_fp, wav3_fp, ffmpeg_exe)

    # Augmentasi visual (aug4, aug5, aug6) untuk setiap frame __s001, __s002, ...
    for img_rel in img_rel_list:
        src_img_fp = os.path.join(out_root, img_rel)
        m = re.match(r'^(.*)(__s\d{3}\.jpg)$', img_rel)
        if not m:
            continue
        prefix_rel, suffix = m.group(1), m.group(2)
        img4_fp = os.path.join(out_root, f'{prefix_rel}__aug4{suffix}')
        img5_fp = os.path.join(out_root, f'{prefix_rel}__aug5{suffix}')
        img6_fp = os.path.join(out_root, f'{prefix_rel}__aug6{suffix}')

        if os.path.exists(img4_fp) and os.path.exists(img5_fp) and os.path.exists(img6_fp):
            continue

        img = cv2.imread(src_img_fp)
        if img is None:
            continue
        if not os.path.exists(img4_fp):
            cv2.imwrite(img4_fp, aug_brightness(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not os.path.exists(img5_fp):
            cv2.imwrite(img5_fp, aug_gaussian_blur(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not os.path.exists(img6_fp):
            aug_video_compression(img, img6_fp)


def main():
    ap = argparse.ArgumentParser(description='Augmentasi DIVINA Skenario B (Opsi B: Train Only)')
    ap.add_argument('--out', default=r'C:\TA2\Dataset\DIVINA_processed',
                    help='Folder root dataset DIVINA_processed')
    ap.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                    help='Folder data tempat divina_labels.csv dan divina_vad.pickle disimpan')
    ap.add_argument('--subsets', nargs='+', default=['train'],
                    help='Subset yang diaugmentasi (default: train saja sesuai Opsi B)')
    ap.add_argument('--workers', type=int, default=8,
                    help='Jumlah thread paralel untuk pemrosesan')
    args = ap.parse_args()

    labels_fp = os.path.join(args.data_dir, 'divina_labels.csv')
    vad_fp = os.path.join(args.data_dir, 'divina_vad.pickle')
    orig_labels_fp = os.path.join(args.data_dir, 'divina_labels_orig.csv')
    orig_vad_fp = os.path.join(args.data_dir, 'divina_vad_orig.pickle')

    assert os.path.exists(labels_fp), f'File label tidak ditemukan: {labels_fp}'
    assert os.path.exists(vad_fp), f'File VAD tidak ditemukan: {vad_fp}'

    # Backup data original murni pada eksekusi pertama agar selalu idempotent
    if not os.path.exists(orig_labels_fp):
        df_init = pd.read_csv(labels_fp)
        if 'augmentation' in df_init.columns:
            df_init = df_init[df_init['augmentation'] == 'orig'].drop(columns=['augmentation'])
        df_init.to_csv(orig_labels_fp, index=False)
        print(f'Backup label original disimpan ke: {orig_labels_fp}')

    if not os.path.exists(orig_vad_fp):
        with open(vad_fp, 'rb') as f:
            vad_init = pickle.load(f)
        vad_orig = {k: v for k, v in vad_init.items() if '__aug' not in k}
        with open(orig_vad_fp, 'wb') as f:
            pickle.dump(vad_orig, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Backup VAD original disimpan ke: {orig_vad_fp}')

    df_orig = pd.read_csv(orig_labels_fp)
    with open(orig_vad_fp, 'rb') as f:
        vad_orig = pickle.load(f)

    target_subsets = set(args.subsets)
    print(f'Subset yang akan diaugmentasi (x7): {sorted(target_subsets)}')

    # Kelompokkan frame per klip pada subset target
    df_target = df_orig[df_orig['subset'].isin(target_subsets)]
    clip_to_imgs = df_target.groupby('audio_file_path')['image_file_path'].apply(list).to_dict()
    print(f'Total klip asli pada {sorted(target_subsets)}: {len(clip_to_imgs)} klip')

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Proses pembuatan file audio & gambar augmentasi secara paralel
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_one_clip, clip_rel, imgs, args.out, ffmpeg_exe)
            for clip_rel, imgs in clip_to_imgs.items()
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Augmentasi klip'):
            fut.result()

    # Bangun tabel label baru & kamus VAD baru
    new_rows = []
    new_vad = dict(vad_orig)

    for clip_rel, group in df_orig.groupby('audio_file_path', sort=False):
        base_records = group.to_dict('records')
        subset_name = base_records[0]['subset']

        # 1) Variasi Original (untuk semua subset: train, devel, test)
        for r in base_records:
            r_copy = dict(r)
            r_copy['augmentation'] = 'orig'
            new_rows.append(r_copy)

        if subset_name not in target_subsets:
            continue

        # 2) Enam variasi augmentasi (aug1 s/d aug6) untuk subset target (train)
        stem_rel = clip_rel[:-4]
        vad_entry = vad_orig.get(clip_rel, [{'start': 0, 'end': SR * 4}])

        # Aug-1, Aug-2, Aug-3: Audio berubah, Video tetap Asli
        for aug_tag in ('aug1', 'aug2', 'aug3'):
            aug_audio_rel = f'{stem_rel}__{aug_tag}.wav'
            new_vad[aug_audio_rel] = vad_entry
            for r in base_records:
                r_copy = dict(r)
                r_copy['audio_file_path'] = aug_audio_rel
                r_copy['augmentation'] = aug_tag
                new_rows.append(r_copy)

        # Aug-4, Aug-5, Aug-6: Audio tetap Asli, Video berubah
        for aug_tag in ('aug4', 'aug5', 'aug6'):
            for r in base_records:
                r_copy = dict(r)
                m = re.match(r'^(.*)(__s\d{3}\.jpg)$', r['image_file_path'])
                if m:
                    r_copy['image_file_path'] = f'{m.group(1)}__{aug_tag}{m.group(2)}'
                r_copy['augmentation'] = aug_tag
                new_rows.append(r_copy)

    df_final = pd.DataFrame(new_rows)
    df_final.to_csv(labels_fp, index=False)
    with open(vad_fp, 'wb') as f:
        pickle.dump(new_vad, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Ringkasan statistik
    u_orig = df_orig.drop_duplicates('audio_file_path')
    u_final = df_final.drop_duplicates(['audio_file_path', 'augmentation'])

    print('\n' + '=' * 65)
    print('RINGKASAN DATASET DIVINA SETELAH AUGMENTASI (OPSI B - TRAIN ONLY)')
    print('=' * 65)
    print('\n1. Jumlah klip SEBELUM augmentasi (Original):')
    print(pd.crosstab(u_orig['subset'], u_orig['occlusion_code'], margins=True))

    print('\n2. Jumlah klip SESUDAH augmentasi (Train x7, Devel & Test tetap Asli):')
    print(pd.crosstab(u_final['subset'], u_final['occlusion_code'], margins=True))

    print('\n3. Rincian per jenis variasi augmentasi pada subset TRAIN:')
    u_train = u_final[u_final['subset'] == 'train']
    print(pd.crosstab(u_train['augmentation'], u_train['occlusion_code'], margins=True))
    print('=' * 65)


if __name__ == '__main__':
    main()
