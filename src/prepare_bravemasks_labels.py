"""
Script to create bravemasks_labels.csv from original BRAVE-MASKS lab_with_stats CSVs.
Reads train.csv, devel.csv, and test.csv in read-only mode and outputs a unified CSV.
"""
import os
import argparse
import pandas as pd


def generate_labels(dataset_dir: str, output_path: str) -> None:
    lab_dir = os.path.join(dataset_dir, 'lab_with_stats')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Reading CSVs from: {lab_dir}")
    train_df = pd.read_csv(os.path.join(lab_dir, 'train.csv'))
    devel_df = pd.read_csv(os.path.join(lab_dir, 'devel.csv'))
    test_df = pd.read_csv(os.path.join(lab_dir, 'test.csv'))

    # 1. Train subset
    train_df['audio_file_path'] = 'audio/train/' + train_df['a_filename']
    train_df['image_file_path'] = 'video/train/' + train_df['v_filename']
    train_df['mask_type'] = train_df['label']
    train_df['subset'] = 'train'
    train_df['original_subset'] = 'train'

    # 2. Devel subset (Used as 'test' for evaluation since it has ground truth labels)
    devel_df['audio_file_path'] = 'audio/devel/' + devel_df['a_filename']
    devel_df['image_file_path'] = 'video/devel/' + devel_df['v_filename']
    devel_df['mask_type'] = devel_df['label']
    devel_df['subset'] = 'test'
    devel_df['original_subset'] = 'devel'

    # 3. Test subset (Confidential labels '?')
    test_df['audio_file_path'] = 'audio/test/' + test_df['a_filename']
    test_df['image_file_path'] = 'video/test/' + test_df['v_filename']
    test_df['mask_type'] = test_df['label']
    test_df['subset'] = 'blind_test'
    test_df['original_subset'] = 'test'

    cols = ['audio_file_path', 'image_file_path', 'gender', 'age', 'mask_type', 'channel', 'subset', 'original_subset']
    combined_df = pd.concat([train_df[cols], devel_df[cols], test_df[cols]], ignore_index=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Successfully generated: {output_path}")
    print(f"Total rows: {len(combined_df):,}")
    print(combined_df['subset'].value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=r'C:\TA2\BRAVE-MASK\shared')
    parser.add_argument('--output_path', type=str, default=r'data/bravemasks_labels.csv')
    args = parser.parse_args()
    generate_labels(args.dataset_dir, args.output_path)

