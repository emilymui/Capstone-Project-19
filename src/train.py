import dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from model import MIMICModule

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

class config:
    def __init__(self):
        self.split='no_test'
        self.num_workers=4
        self.batch_size=32
        '''
        Key is dataframe name, should match name of attribute in dataset class,
        Which in turn should be IDENTICAL to the actual name of the csv
        Value is list of columns to use
        
        Can use this for reading data like so:
        for module in first set of keys:
            for csv in inner set of keys:
                read data, use module[csv] as usecols argument
        '''
        self.data_to_use = {
            'core': {
                'admissions': ['subject_id','ethnicity','insurance','admission_location',
                              'discharge_location'],
                'patients': ['subject_id','gender','anchor_age'],
                'transfers': ['subject_id','eventtype', 'careunit']
            },
            'ed': {
                'diagnosis': ['subject_id','icd_code','icd_version'],
                'vitalsign': ['subject_id', 'temperature', 'heartrate',
                              'resprate', 'o2sat', 'sbp', 'dbp','rhythm',
                              'pain']                
            },
            'hosp': {
                'diagnoses_icd': ['subject_id','icd_code','icd_version','seq_num'],
                'drgcodes': ['subject_id', 'drg_type', 'drg_code', 'drg_severity',
                             'drg_mortality'],
                'services': ['subject_id', 'curr_service'],
                'microbiologyevents': ['subject_id', 'spec_itemid', 'spec_type_desc', 'ab_itemid',
                                       'test_itemid', 'test_name', 'org_itemid', 'org_name', 
                                       'dilution_value', 'interpretation'],
                'emar': ['subject_id', 'medication'],
                'labevents': ['subject_id', 'flag','priority','itemid'],
                'pharmacy': ['subject_id', 'medication','status'],
                'poe': ['subject_id', 'order_type'],
                'prescriptions': ['subject_id', 'gsn'],
                'procedures_icd': ['subject_id', 'icd_code', 'icd_version']
            }
        }
        self.prepared_data=True
callbacks = [
    ModelCheckpoint(monitor='val_total_loss', every_n_epochs=1, save_last=True),
    GPUStatsMonitor()
]

cfg = config()
strat = 'dp' if torch.cuda.device_count() < 2 else 'ddp'
#print(torch.cuda.device_count(), torch.cuda.is_available(), torch.version.cuda, torch.backends.cudnn.enabled, torch.backends.cudnn.version())
#print(torch.cuda.device_count())
print(start, torch.cuda.device_count(), torch.cuda.is_available(), torch.version.cuda)
trainer = pl.Trainer(default_root_dir='/scratch/arz8448/capstone/notebooks/outputs/', gpus=torch.cuda.device_count(),
                     accelerator='auto', strategy=strat, precision=16, callbacks=callbacks,
                     check_val_every_n_epoch=1, max_epochs=12)
data = dataset.MIMICDataModule(cfg)
print('Begin data setup')
data.setup()
print('End data setup')
col_idx, cont_cols, embed_in = data.get_features_for_tabular()
model = MIMICModule(cfg, col_idx, cont_cols, embed_in)
print('Begin training')
trainer.fit(model, datamodule=data)
print('End training')