"""
Preprocessing dataset DIVINA agar formatnya sama dengan BRAVE-MASKS (input ORAGEN).

Alur per video (contoh: M01_helm.mp4):
  1. Ekstrak audio -> WAV 16 kHz mono (ffmpeg).
  2. Silero VAD pada audio penuh -> segmen ucapan; segmen digabung/dipotong menjadi klip
     setingkat kalimat (durasi 1-12 detik).
  3. Setiap klip disimpan sebagai audio/<split>/<ID>_<oklusi>_<nnn>.wav
  4. Setiap detik klip diambil 1 frame (sesuai ORAGEN: 1 frame/detik), wajah dideteksi dengan
     OpenCV YuNet, lalu dipotong (kepala + aksesoris oklusi) dan di-resize 224x224:
     video/<split>/<ID>_<oklusi>_<nnn>__s001.jpg, __s002.jpg, ...
  5. Baris label (1 baris per frame, sama seperti bravemasks_labels.csv) + VAD per klip.

Output:
  <OUT_ROOT>/audio/{train,devel,test}/*.wav
  <OUT_ROOT>/video/{train,devel,test}/*.jpg
  data/divina_labels.csv, data/divina_vad.pickle, data/divina_preprocess_report.csv

Video asli TIDAK diubah (hanya dibaca).
Jalankan:  python src/prepare_divina.py            (semua video)
           python src/prepare_divina.py --only F01_hijab   (uji 1 video)
"""
import os
import re
import glob
import json
import math
import pickle
import argparse
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
import imageio_ffmpeg

SR = 16000

# ---------------- Metadata partisipan ----------------
PARTICIPANTS = {
    'M01': {'gender': 'male', 'age': 22}, 'M02': {'gender': 'male', 'age': 35},
    'M03': {'gender': 'male', 'age': 48}, 'M04': {'gender': 'male', 'age': 55},
    'F01': {'gender': 'female', 'age': 21}, 'F02': {'gender': 'female', 'age': 36},
    'F03': {'gender': 'female', 'age': 46}, 'F04': {'gender': 'female', 'age': 50},
}
# Speaker-independent split 2:1:1 per gender
SPLIT = {'M01': 'train', 'M04': 'train', 'F01': 'train', 'F04': 'train',
         'M02': 'devel', 'F03': 'devel',
         'M03': 'test', 'F02': 'test'}
# Nama file -> label oklusi (nama kelas sama dengan C_NAMES 'mask' 10 kelas di config)
OCCLUSION = {'polos': ('No mask', 'NO'), 'hijab': ('Hijab', 'HJ'), 'helm': ('Helmet', 'HM'),
             'kacamata': ('Glasses', 'KA'), 'topi': ('Hat', 'TP')}

# ---------------- Parameter segmentasi ----------------
MIN_CLIP, MAX_CLIP = 1.0, 12.0     # detik (WIN_MIN_LENGTH ORAGEN = 1 detik)
MERGE_GAP = 0.25                   # gabungkan segmen VAD yang jedanya < 0.25 s
PAD = 0.15                         # padding sebelum/sesudah ucapan (detik)


def ffmpeg_extract_audio(video_fp, wav_fp):
    os.makedirs(os.path.dirname(wav_fp), exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video_fp,
           '-vn', '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_s16le', wav_fp]
    subprocess.run(cmd, check=True)


def load_vad():
    local_dir = os.path.expanduser(r'~\.cache\torch\hub\snakers4_silero-vad_master')
    if os.path.exists(local_dir):
        model, utils = torch.hub.load(local_dir, 'silero_vad', source='local', trust_repo=True, verbose=False)
    else:
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, verbose=False)
    return model, utils[0]  # get_speech_timestamps


def make_clips(wav, vad_model, get_ts):
    """Segmen VAD -> klip setingkat kalimat (detik)."""
    ts = get_ts(wav, vad_model, sampling_rate=SR, min_silence_duration_ms=300,
                min_speech_duration_ms=250, speech_pad_ms=0)
    segs = [[t['start'] / SR, t['end'] / SR] for t in ts]
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < MERGE_GAP and e - merged[-1][0] <= MAX_CLIP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    total = len(wav) / SR
    clips = []
    for s, e in merged:
        s, e = max(0.0, s - PAD), min(total, e + PAD)
        # potong segmen yang terlalu panjang menjadi beberapa bagian sama rata
        n = max(1, math.ceil((e - s) / MAX_CLIP))
        step = (e - s) / n
        for i in range(n):
            cs, ce = s + i * step, s + (i + 1) * step
            if ce - cs >= MIN_CLIP:
                clips.append((round(cs, 3), round(ce, 3)))
    return clips


class FaceCropper:
    def __init__(self, model_fp):
        self.det = cv2.FaceDetectorYN.create(model_fp, '', (320, 320), score_threshold=0.6)

    def detect(self, frame):
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            scale = 1.0
            small = frame
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda x: x[2] * x[3])  # wajah terbesar
        x, y, bw, bh = (f[:4] / scale).tolist()
        return [x, y, bw, bh]

    @staticmethod
    def crop(frame, box, size=224):
        """Crop persegi: wajah + area kepala agar aksesoris (helm/topi/hijab) ikut terlihat."""
        x, y, bw, bh = box
        side = 1.6 * max(bw, bh)
        cx, cy = x + bw / 2, y + bh / 2 - 0.12 * bh  # geser sedikit ke atas (topi/helm)
        H, W = frame.shape[:2]
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
        pad = [max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W)]
        if any(pad):
            frame = cv2.copyMakeBorder(frame, *pad, cv2.BORDER_REPLICATE)
            x0, x1, y0, y1 = x0 + pad[2], x1 + pad[2], y0 + pad[0], y1 + pad[0]
        return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def read_frames_at(video_fp, times):
    """Baca frame pada daftar waktu (detik) menggunakan ffmpeg pipe @ 2 fps (3.7x lebih cepat dari OpenCV grab)."""
    cap = cv2.VideoCapture(video_fp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    pipe_fps = 2.0
    pw, ph = 960, int(round(960 * h / w))
    if ph % 2 == 1:
        ph += 1

    target_bins = sorted({max(0, int(round(t * pipe_fps))) for t in times})
    max_bin = target_bins[-1] if target_bins else 0
    target_set = set(target_bins)

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-hwaccel', 'auto', '-loglevel', 'error',
        '-i', video_fp, '-vf', f'fps={pipe_fps},scale={pw}:{ph}',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = pw * ph * 3
    bin_frames = {}
    b_idx = 0
    last_frame = None
    while True:
        buf = p.stdout.read(frame_bytes)
        if len(buf) != frame_bytes:
            break
        if b_idx in target_set or last_frame is None:
            fr = np.frombuffer(buf, dtype=np.uint8).reshape((ph, pw, 3)).copy()
            last_frame = fr
            if b_idx in target_set:
                bin_frames[b_idx] = fr
        b_idx += 1
    p.stdout.close()
    p.wait()

    out = {}
    for t in times:
        fi = min(max(0, n_frames - 1), int(round(t * fps)))
        b = max(0, int(round(t * pipe_fps)))
        if b in bin_frames:
            out[fi] = bin_frames[b]
        elif last_frame is not None:
            out[fi] = last_frame
    return out, fps, n_frames


def process_video(video_fp, out_root, tmp_root, vad_model, get_ts, cropper):
    stem = os.path.splitext(os.path.basename(video_fp))[0]          # M01_helm
    pid, occ = stem.split('_', 1)
    split = SPLIT[pid]
    mask_type, occ_code = OCCLUSION[occ]

    full_wav = os.path.join(tmp_root, stem + '.wav')
    if not os.path.exists(full_wav):
        ffmpeg_extract_audio(video_fp, full_wav)
    wav, sr = torchaudio.load(full_wav)
    wav = wav.mean(0)
    assert sr == SR

    clips = make_clips(wav, vad_model, get_ts)

    # waktu frame: detik ke-k klip -> titik tengah detik itu
    frame_req = []
    for ci, (cs, ce) in enumerate(clips):
        for k in range(1, math.ceil(ce - cs) + 1):
            t = cs + min(k - 0.5, (ce - cs) - 0.02)
            frame_req.append((ci, k, t))
    frames, fps, n_frames = read_frames_at(video_fp, [t for _, _, t in frame_req])

    # deteksi wajah; fallback = median bbox dari frame yang terdeteksi pada video yang sama
    boxes = {}
    for ci, k, t in frame_req:
        fi = min(n_frames - 1, int(round(t * fps)))
        if fi in frames and fi not in boxes:
            boxes[fi] = cropper.detect(frames[fi])
    valid = np.array([b for b in boxes.values() if b is not None])
    median_box = np.median(valid, axis=0).tolist() if len(valid) else None

    a_dir = os.path.join(out_root, 'audio', split)
    v_dir = os.path.join(out_root, 'video', split)
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    rows, vad, n_fallback, n_imgs = [], {}, 0, 0
    for ci, (cs, ce) in enumerate(clips):
        clip_name = f'{stem}_{ci + 1:03d}'
        a_rel = f'audio/{split}/{clip_name}.wav'
        start_idx, end_idx = int(cs * SR), int(ce * SR)
        seg = wav[start_idx:end_idx].unsqueeze(0)
        torchaudio.save(os.path.join(out_root, a_rel), seg, SR, encoding='PCM_S', bits_per_sample=16)
        # Klip sudah merupakan segmen ucapan aktif hasil Silero VAD dari audio penuh
        vad[a_rel] = [{'start': 0, 'end': end_idx - start_idx}]

        for (cj, k, t) in [r for r in frame_req if r[0] == ci]:
            fi = min(n_frames - 1, int(round(t * fps)))
            fr = frames.get(fi)
            if fr is None:
                continue
            box = boxes.get(fi)
            detected = box is not None
            if not detected:
                box = median_box
                n_fallback += 1
            if box is None:
                continue
            v_rel = f'video/{split}/{clip_name}__s{k:03d}.jpg'
            cv2.imwrite(os.path.join(out_root, v_rel), FaceCropper.crop(fr, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_imgs += 1
            rows.append({
                'audio_file_path': a_rel, 'image_file_path': v_rel,
                'gender': PARTICIPANTS[pid]['gender'], 'age': PARTICIPANTS[pid]['age'],
                'mask_type': mask_type, 'channel': 'c', 'subset': split, 'original_subset': split,
                'participant_id': pid, 'occlusion_code': occ_code,
                'clip_start': cs, 'clip_end': ce, 'face_detected': int(detected),
            })

    report = {'video': stem, 'participant': pid, 'split': split, 'occlusion': occ_code,
              'video_duration_s': round(n_frames / fps, 1), 'n_clips': len(clips),
              'speech_duration_s': round(sum(e - s for s, e in clips), 1),
              'n_images': n_imgs, 'face_detect_rate': round(1 - n_fallback / max(1, n_imgs), 4)}
    return rows, vad, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r'C:\TA2\Dataset\DIVINA')
    ap.add_argument('--out', default=r'C:\TA2\Dataset\DIVINA_processed')
    ap.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    ap.add_argument('--yunet', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    'pretrained_models', 'face_detection_yunet_2023mar.onnx'))
    ap.add_argument('--only', default=None, help='proses 1 video saja, mis. F01_hijab')
    args = ap.parse_args()

    videos = sorted(glob.glob(os.path.join(args.src, '**', '*.mp4'), recursive=True))
    videos = [v for v in videos if re.match(r'^[MF]\d{2}_(polos|hijab|helm|kacamata|topi)$', os.path.splitext(os.path.basename(v))[0])]
    if args.only:
        videos = [v for v in videos if os.path.splitext(os.path.basename(v))[0] == args.only]
    print(f'Video ditemukan: {len(videos)}')

    tmp_root = os.path.join(args.out, '_full_audio')
    part_root = os.path.join(args.out, '_parts')      # hasil per video -> bisa dilanjutkan (resume)
    os.makedirs(part_root, exist_ok=True)
    vad_model, get_ts = load_vad()
    cropper = FaceCropper(args.yunet)

    for i, v in enumerate(videos, 1):
        stem = os.path.splitext(os.path.basename(v))[0]
        part_fp = os.path.join(part_root, stem + '.pkl')
        if os.path.exists(part_fp):
            print(f'[{i}/{len(videos)}] {stem}: sudah diproses, dilewati')
            continue
        rows, vad, report = process_video(v, args.out, tmp_root, vad_model, get_ts, cropper)
        with open(part_fp, 'wb') as f:
            pickle.dump({'rows': rows, 'vad': vad, 'report': report}, f)
        print(f'[{i}/{len(videos)}] {json.dumps(report)}', flush=True)

    # gabungkan semua hasil per video
    rows, vad, reports = [], {}, []
    for fp in sorted(glob.glob(os.path.join(part_root, '*.pkl'))):
        d = pickle.load(open(fp, 'rb'))
        rows += d['rows']; vad.update(d['vad']); reports.append(d['report'])
    os.makedirs(args.data_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.data_dir, 'divina_labels.csv'), index=False)
    with open(os.path.join(args.data_dir, 'divina_vad.pickle'), 'wb') as f:
        pickle.dump(vad, f, protocol=pickle.HIGHEST_PROTOCOL)
    rep = pd.DataFrame(reports)
    rep.to_csv(os.path.join(args.data_dir, 'divina_preprocess_report.csv'), index=False)
    print('\n', rep.to_string(index=False))
    if len(df):
        u = df.drop_duplicates('audio_file_path')
        print('\nJumlah klip per split x oklusi:\n', pd.crosstab(u['subset'], u['occlusion_code'], margins=True))


if __name__ == '__main__':
    main()
