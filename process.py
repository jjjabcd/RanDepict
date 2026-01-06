import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import warnings
import argparse

warnings.filterwarnings('ignore')

# Suppress all logging completely
import logging
logging.getLogger().setLevel(logging.CRITICAL)


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import rdkit
rdkit.rdBase.DisableLog('rdApp.error')
rdkit.rdBase.DisableLog('rdApp.warning')
rdkit.rdBase.DisableLog('rdApp.info')

import sys
from io import StringIO

class SuppressOutput:
    def __init__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

from RanDepict import RandomDepictor


class CustomRandomDepictor(RandomDepictor):
    def __init__(self, *args, **kwargs):
        super(CustomRandomDepictor, self).__init__(*args, **kwargs)
        self.PATH_BKG = self.HERE.joinpath("backgrounds/")
        self.BKGS = [
            'bg-1041.png', 'bg-12.png', 'bg-167.png', 'bg-27.png', 'bg-378.png',
            'bg-46.png', 'bg-468.png', 'bg-635.png', 'bg-664.png', 'bg-76.png', 'bg-799.png'
        ]

    def to_hand_written(self, depiction):
        mol_aug = self.hand_drawn_augment(depiction)
        background_selected = self.random_choice(self.BKGS)
        bkg = cv2.imread(os.path.join(os.path.normpath(self.PATH_BKG), background_selected))
        if bkg is None:
            return mol_aug
        bkg = cv2.resize(bkg, (384, 384))
        p = 0.7
        mol_bkg = cv2.addWeighted(mol_aug, p, bkg, 1 - p, gamma=0)
        return mol_bkg

    def hand_drawn_augment(self, img) -> np.array:
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.resize_hd(img)
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.erode(img)
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.aspect_ratio(img, "mol")
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.affine(img, "mol")
        if img.shape[:2] != (384, 384):
            img = cv2.resize(img, (384, 384))
        return img

    def process_smiles(self, smiles):
        try:
            with SuppressOutput():
                depiction = self(smiles)
                return self.to_hand_written(depiction)
        except:
            return None


def process_dataset(dataset_name, df, depictor, output_dir, task_name, k=1):
    # Create images directory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    target_col = df.columns[-1]
    results = []
    
    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
        smiles = row['SMILES_deepchem']
        image_id = row['IDs_deepchem']
        
        # Generate image
        augmented_image = depictor.process_smiles(smiles)
        
        if augmented_image is not None:
            # Save image
            img = Image.fromarray(augmented_image)
            img.save(os.path.join(images_dir, f'{image_id}.png'))
            
            # Generate file path
            randepict_file_path = f"../data/Classification/{task_name}/Randepict/images/{image_id}.png"
            
            # Generate CSV row
            if dataset_name == "test":
                # For test.csv: two file_path columns
                decimer_file_path = ""
                if 'IDs_decimer' in df.columns and pd.notna(row['IDs_decimer']) and str(row['IDs_decimer']).strip():
                    decimer_file_path = f"../data/Classification/{task_name}/test_images/{str(row['IDs_decimer']).strip()}.png"
                
                result_row = {
                    'IDs_deepchem': row.get('IDs_deepchem', ''),
                    'randepict_file_path': randepict_file_path,
                    'decimer_file_path': decimer_file_path,
                    target_col: row[target_col],
                    'IDs_decimer': row.get('IDs_decimer', ''),
                    'SMILES_deepchem': row.get('SMILES_deepchem', row.get('SMILES', '')),
                    'SMILES_decimer': row.get('SMILES_decimer', row.get('SMILES', ''))
                }
            else:
                # train.csv용: 기존 file_path 컬럼
                result_row = {
                    'IDs_deepchem': row.get('IDs_deepchem', ''),
                    'file_path': randepict_file_path,
                    target_col: row[target_col],
                    'IDs_decimer': row.get('IDs_decimer', ''),
                    'SMILES_deepchem': row.get('SMILES_deepchem', row.get('SMILES', '')),
                    'SMILES_decimer': row.get('SMILES_decimer', row.get('SMILES', ''))
                }
            
            results.append(result_row)
    
    # CSV 파일 저장
    if results:
        result_df = pd.DataFrame(results)
        csv_output_path = os.path.join(output_dir, f'{dataset_name}_dataset.csv')
        result_df.to_csv(csv_output_path, index=False)
        print(f"✓ {dataset_name} dataset completed: {len(results)} samples")
    
    return csv_output_path if results else None


def main():
    # Java 메모리 설정 및 로깅 차단
    os.environ['JAVA_OPTS'] = '-Xmx2g -Xms1g -Djava.util.logging.config.file=/dev/null'
    os.environ['_JAVA_OPTIONS'] = '-Xmx2g -Xms1g -Djava.util.logging.config.file=/dev/null'
    
    # CDK/RDKit 관련 환경변수 설정
    os.environ['CDK_LOG_LEVEL'] = 'OFF'
    os.environ['RDKIT_LOG_LEVEL'] = 'OFF'
    
    parser = argparse.ArgumentParser(description='Process molecular data with RanDepict')
    parser.add_argument('--task_name', type=str, default='clintox+FDA_APPROVED',
                       help='Task name for data directory')
    parser.add_argument('--k', type=int, default=1,
                       help='Number of augmentations per molecule')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed')
    parser.add_argument('--test_only', action='store_true',
                       help='Process only test dataset')
    parser.add_argument('--train_only', action='store_true',
                       help='Process only train dataset')
    
    args = parser.parse_args()
    
    print(f"RanDepict Processing - Task: {args.task_name}")
    
    # 경로 설정
    task_name = args.task_name
    data_dir = os.path.join('..', '..', 'data', 'Classification', task_name)
    output_dir = os.path.join(data_dir, 'Randepict')
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    datasets = []
    
    if not args.test_only:
        train_csv_path = os.path.join(data_dir, 'train.csv')
        if os.path.exists(train_csv_path):
            train_df = pd.read_csv(train_csv_path)
            datasets.append(('train', train_df))
            print(f"Loaded train dataset: {len(train_df)} samples")
    
    if not args.train_only:
        test_csv_path = os.path.join(data_dir, 'test.csv')
        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            datasets.append(('test', test_df))
            print(f"Loaded test dataset: {len(test_df)} samples")
    
    if not datasets:
        print("No datasets found!")
        return
    
    # RanDepict 초기화
    depictor = CustomRandomDepictor(seed=args.seed, hand_drawn=True)
    print("✓ RanDepict initialized")
    
    # 각 데이터셋 처리
    for dataset_name, df in datasets:
        process_dataset(dataset_name, df, depictor, output_dir, task_name, args.k)
    
    print("Processing completed!")


if __name__ == "__main__":
    main()