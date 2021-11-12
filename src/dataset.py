import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from transformers import AutoTokenizer
from pytorch_widedeep.preprocessing import TabPreprocessor
import random
from numbers import Number
from functools import partial
import pytorch_lightning as pl
from typing import Optional

cxr_dir = '../data/mimic-cxr-jpg/'
mimic_dir = '../data/mimic-iv/'
# This is ALL columns, need to trim it
cat_cols = ['admission_location', 'discharge_location', 'insurance', 'ethnicity',
            'gender', 'anchor_age', 'eventtype', 'careunit', 'icd_code',
            'icd_version', 'rhythm', 'pain', 'seq_num',
            'icd_code_diagnoses_icd', 'icd_version_diagnoses_icd', 'drg_type',
            'drg_code', 'drg_severity', 'drg_mortality', 'curr_service',
            'spec_itemid', 'spec_type_desc', 'test_itemid', 'test_name',
            'org_itemid', 'org_name', 'ab_itemid', 'dilution_value',
            'interpretation', 'medication', 'itemid', 'flag', 'priority',
            'medication_pharmacy', 'status', 'order_type', 'gsn','icd_code_procedures_icd',
            'icd_version_procedures_icd']
cont_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']

# These should eventually live in an augmentations.py module
class GaussianNoise(object):
    def __init__(self, mean=0., std=1., p=1.):
        self.mean = mean
        self.std = std
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(self.std, (tuple, list)):
                std = np.random.uniform(self.std[0], self.std[1])
            else:
                std = self.std
            return torch.clip(img + (torch.randn(img.size())*std + self.mean), 0., 1.)
        else:
            return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class GrayscaleJitter(object):
    '''
    A class that implements grayscale jittering, for use with torchvision.transforms.Compose
    
    Parameters:
        brightness_scale: If numeric, value by which to increase/decrease pixel values
                          If iterable, will adjust by random float in range [min, max]
        contrast_scale: If float, ratio by which to multiply pixels
                        If iterable, will multiple by random float in range [min, max]
        p (float): Probability of applying the transformation. Default = 1.0
    '''
    def __init__(self, brightness, contrast, p=1.0):
        self._check_input(brightness, 'brightness')
        self._check_input(contrast, 'contrast')
        if not 0.0 <= p and 1.0 >= p:
            raise ValueError('p should be in range [0, 1]')
        self.brightness = brightness
        self.contrast = contrast

        self.p = p
        
    def _check_input(self, value, name):
        if isinstance(value, Number):
            if value < 0.0:
                raise ValueError(f'{name} must be > 0')
        elif isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f'If tuple/list, {name} must be of len 2')
            mn, mx = value
            if not mn < mx:
                raise ValueError(f'Min of {name} should be less than max')
            if name == 'contrast' and mn < 0:
                raise ValueError(f'All elements of {name} must be > 0')
        else:
            raise TypeError(f'{name} must be of type float, or tuple/list with len 2')
        
    def __call__(self, img):
        '''
        Adjusts img by brightness_scale and contrast_scale with probability p

        Args:
            img: The image as a tensor in the range [0, 1], i.e. after dividing by 255
                 but before normalizing.
        '''
        if random.random() < self.p:
            if isinstance(self.brightness, (tuple, list)):
                # If float, we treat
                b = random.uniform(self.brightness[0], self.brightness[1])
            else:
                b = self.brightness
                
            if isinstance(self.contrast, (tuple, list)):
                c = random.uniform(self.contrast[0], self.contrast[1])
            else:
                c = self.contrast
            return torch.clip(img*c + b, 0., 1.)
        
        else:
            return image
        
    # Can be modified to also print additional info about the class
    def __repr__(self):
        return self.__class__.__name__ + (f'brightness={self.brightness}, '+
                                          f'contrast={self.contrast}, p={self.p}')
class MIMICDataset(Dataset):
    '''
    Dataset encapsulating all multi-modal data available for patients
    in the MIMIC_CXR-JPG dataset
    '''
    
    def __init__(self, config, split, aug=None, image_size=(224,224),
                 cxr_dir='../data/mimic-cxr-jpg/',
                 records_csv='cxr-record-list.csv', mimic_dir='../data/mimic-iv/',
                 split_csv='mimic-cxr-2.0.0-split.csv', indexed_vocab=True,
                 tab_preprocessor=None, prepared_data=False):
        
        super().__init__()
        
        self.cxr_dir = cxr_dir
        self.mimic_dir = mimic_dir
        
        assert split in ('train','val','validation','validate','test','all')
        self.split = split if split not in ['validation','val'] else 'validate'
        self.augs = None # Add augmentations later
        
        # Verified that splits csv has some ordering of images as records csv, 
        # so this is a safe way to filter to the split
        records = cxr_dir+records_csv
        splits = cxr_dir+split_csv
        if self.split == 'all':
            self.records = pd.read_csv(records)
        else:
            self.records = pd.read_csv(records)[pd.read_csv(splits).split==self.split]
        self.our_subjects = self.records.subject_id.unique()
        
        # Read in and preprocess the tabular data
        if prepared_data:
            self.data = pd.read_csv(f'../data/saved/{self.split}_most_recent_2.csv',
                                    index_col='subject_id')
        else:
            self.data = self.read_data(config.data_to_use)
        self.data = self.data[cat_cols+cont_cols]
            
        if not tab_preprocessor:
            self.tab_preprocessor = TabPreprocessor(embed_cols=cat_cols,
                                                    continuous_cols=cont_cols, 
                                                    for_transformer=True,
                                                    default_embed_dim=16)
            self.tab_preprocessor.fit(self.data)
        else: # Should pass the tab preprocessor fit on the train data
            self.tab_preprocessor = tab_preprocessor
        
        self.image_size = image_size
        if self.split in ['train', 'all']:
            # Transforms used are based on SimCLR:
            # https://github.com/sthalles/SimCLR/tree/
            ks = int(0.1*self.image_size[0])
            ks = ks - ((ks+1)%2)
            
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine((-20, 20),
                                        translate=(0.1, 0.1),
                                        scale=(0.95, 1.05)),
                #GrayscaleJitter([-0.15, 0.15], [0.9,1.1]),
                transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6,1.4)),
                #GaussianNoise(0., (0.1, 3.0)),
                transforms.GaussianBlur(kernel_size=ks, sigma=(0.1,3.0))
            ])
                
        else:
            self.transforms = nn.Identity()
        
    def __len__(self):
        return self.records.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        subset = self.records.iloc[idx]
        subject = subset['subject_id']
        
        # Image handling
        image = ImageOps.grayscale(Image.open(self.cxr_dir+subset['path'][:-3]+'jpg'))
        image = self.transforms(image) 
        image = self.preprocess_image(image)
        
        
        # Text handling
        text_path = self.cxr_dir+subset['path'][:29]+'.txt'
        if os.path.exists(text_path):
            with open(text_path) as f:
                text = f.read()
        else:
            text = np.array([''])
            
        # Tabular handling - preprocessing already done
        tabular = self.data.loc[subject]
        
        return {'image': image, 'text': text, 'tabular': tabular}
    
    def preprocess_image(self, image):
        # Uses weighted mean and std from MIMIC-CXR-JPG, Stanford, NIH Chest X-ray data:
        normalizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.4860, std=0.2874)
        ])
        image = normalizer(image)
        
        # Downsampling to make all images the same size
        # Note that interpolate wants images in shape (batch_size, num_channels, *additional_dims)
        image = F.interpolate(image.unsqueeze(0), size=self.image_size).squeeze(0)
        return image
    
    def read_data(self, data_dict):
        # By the time this function finishes, every subject should have ONE row
        data = pd.DataFrame(index = self.records.subject_id.unique())
        for module in data_dict.keys(): # Iteration over modules
            for csv in data_dict[module].keys(): # Iteration over csvs within module
                agg_csv = partial(self.mimic_agg, csv=csv)
                df = pd.read_csv(f'{mimic_dir}{module}/{csv}.csv',
                                 index_col = 'subject_id',
                                 usecols = data_dict[module][csv])
                df = df[df.index.isin(self.our_subjects)] 
                # Aggregation should happen below these comments
                # One row per subject per table. Likely requires different logic per table...
                
                #df = df.groupby(df.index).apply(agg_csv)
                
                # For now just using most recent data for everyone
                df = df.groupby(df.index).tail(1)
                data = data.join(df, rsuffix=f'_{csv}')
        return data
    
    def mimic_agg(self, x, csv):
        '''
        by_latest = []
        if csv in by_latest: # Return last row
            return x.iloc[-1]
        else: # Other logic
            return ??
        '''
        pass

    def get_features_for_tabular(self):
        '''
        Returns the columnx_idx and continuous_cols from the widedeep preprocessor
        for use with the module's TabTransformer
        '''
        col_idx = {k:v for v,k in enumerate(self.data.columns)}
        cont_cols = self.tab_preprocessor.continuous_cols
        embed_in = self.tab_preprocessor.embeddings_input
        return col_idx, cont_cols, embed_in
    
def collate_mimic(batch, tokenizer, tab_preprocessor):
    image = torch.stack([item['image'] for item in batch])
    text = [item['text'] for item in batch]
    tabular = [item['tabular'] for item in batch]
    
    # Alt: Turn batch from dict of lists to list of dicts. Time the difference
    # batch = {k: [d[k] for d in batch] for k in batch[0]}
    # image, text = batch['image'], batch['text']
    
    # Pad all text vectors so same length with <PAD> tokens
    # Tokenize in collate, not getitem, for proper padding
    encoding = tokenizer(text, return_tensors='pt',
                         padding=True, truncation=True)
    text = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    
    # Turn tabular into a DataFrame for easier manipulation
    # May not be needed with widedeep
    col_idx = [(c, i) for i, c in enumerate(tabular[0].index)]
    cat = [i[1] for i in col_idx if i[0] in cat_cols]
    cont = [i[1] for i in col_idx if i[0] not in cat_cols]
    tabular = pd.DataFrame(tabular)
    tabular.iloc[:,cont] = tabular.iloc[:,cont].fillna(0)
    tabular = tab_preprocessor.transform(tabular)
    tabular = np.nan_to_num(tabular, nan=19996)
    tabular = torch.tensor(tabular)
    return image, text, attention_masks, tabular

class MIMICDataModule(pl.LightningDataModule):
    '''
    Parameters:
        split: bool or iterable. If True, creates train/test/val dataloaders;
            If False, creates a single dataloader for entire dataset;
            If iterable, creates only the dataloaders specified
        batch_size: Batch size
        num_workers: Number of workers
        tokenizer: The tokenizer to use for BERT
    '''
    def __init__(self, config):
        super().__init__()
        assert config.split in ['no_test', True, False], "config.split must be one of 'no_test', True, False"
        self.config = config
        self.split = config.split
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",
                                                       model_max_length=512)
        self.prepared_data = config.prepared_data
        
    def setup(self, stage:Optional[str]=None) -> None:
        if self.split == 'no_test':
            self.train_data = MIMICDataset(config=self.config, split='train',
                                           prepared_data=self.prepared_data)
            self.tab_preprocessor = self.train_data.tab_preprocessor
            self.val_data = MIMICDataset(config=self.config, split='validate',
                                         tab_preprocessor=self.tab_preprocessor,
                                         prepared_data=self.prepared_data)
        
        elif not self.split:
            self.train_data = MIMICDataset(config=self.config, split='all')
            self.tab_preprocessor = self.train_data.tab_preprocesor
        
        else:
            self.train_data = MIMICDataset(config=self.config, split='train')
            self.tab_preprocessor = self.train_data.tab_preprocessor
            self.val_data = MIMICDataset(config=self.config, split='validate',
                                         tab_preprocessor = self.tab_preprocessor)
            self.test_data = MIMICDataset(config=self.config, split='test',
                                         tab_preprocessor = self.tab_preprocessor)
        
        self.col_names = self.train_data.data.columns
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=partial(collate_mimic, tokenizer=self.tokenizer,
                                             tab_preprocessor = self.tab_preprocessor))
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=partial(collate_mimic, tokenizer=self.tokenizer,
                                             tab_preprocessor = self.tab_preprocessor))
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=partial(collate_mimic, tokenizer=self.tokenizer,
                                             tab_preprocessor = self.tab_preprocessor))

    def get_features_for_tabular(self):
        '''
        Returns which features are categorical and continuous for use with a
        pytorch-widedeep model, as well as number of unique values for embed column
        '''
        # First term creates column name:number index from data's column names
        col_idx = {k:v for v,k in enumerate(self.col_names)}
        cont_cols = self.tab_preprocessor.continuous_cols
        embed_in = self.tab_preprocessor.embeddings_input
        return col_idx, cont_cols, embed_in
                
                