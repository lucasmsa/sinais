from PIL import Image
import os
import glob
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
# Cross validation foi trocada por model_selection devido a versao 
# usada do sklearn, no final servira ao mesmo proposito
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import scipy.fftpack as ft