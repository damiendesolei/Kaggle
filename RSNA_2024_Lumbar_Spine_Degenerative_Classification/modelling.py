# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pydicom as dicom
import pydicom
import json
import glob
import collections
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import random
from glob import glob


path = 'G:\\kaggle\\rsna-2024-lumbar-spine-degenerative-classification\\'

label_coordinates_df = pd.read_csv( path + 'train_label_coordinates.csv')
train_series = pd.read_csv( path + 'train_series_descriptions.csv')
df_train = pd.read_csv( path + 'train.csv')
df_sub = pd.read_csv( path + 'sample_submission.csv')
test_series = pd.read_csv( path + 'test_series_descriptions.csv')


folder_path = 'G:\\kaggle\\rsna-2024-lumbar-spine-degenerative-classification\\train_images\\100206310\\1012284084\\'
dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
