from PIL import Image
import numpy as np
import scipy.fftpack as ft


image_to_frequency = lambda data: ft.rfft(ft.rfft(data, axis=0), axis=1)

data = {}
frequency = {}

for s_value in range(1, 41):

    data[s_value] = []

    frequency[s_value] = []

    for img_value in range(1, 11):

        new_freq = np.array(Image.open(f'orl_faces/s{s_value}/{img_value}.pgm'))

        data[s_value].append(new_freq)

        frequency[s_value].append(image_to_frequency(data[s_value][img_value-1]))


