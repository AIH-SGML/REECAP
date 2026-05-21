import os
import glob
import sys
from EyeQ.EyeQ_preprocess import EyeQ_process_main

left_eye = '../data/original_imgs' 
prep_images_dir = '../data/imgs'

image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))  
EyeQ_process_main.process(image_paths, output_dir, img_size = (256,256))