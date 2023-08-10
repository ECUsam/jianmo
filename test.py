import gc

from dataset import *
from config import DefaultConfig
from feature import get_feat
from utils import get_label
import warnings
warnings.filterwarnings('ignore')

opt = DefaultConfig()
opt.update()
train1 ,train2 = get_train_data(opt)
train_sample = get_sample(train1, train2)
train_feat = get_feat(train1, train_sample)
train_all = get_label(train_feat, opt)
print(train_all)
gc.collect()
