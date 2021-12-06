import netCDF4 as nc4
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import sys
import glob
sys.path.append("src/")
from nc_tile_extractor import extract_cloudy_labelled_tiles
import zipfile
import lightgbm as lgb
from loader import CumuloDataset
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")
