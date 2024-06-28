import cv2
import numpy as np
import os
import random
from tqdm import tqdm

folders = [
    'subjects_0-1999_72_imgs', 
    'subjects_2000-3999_72_imgs', 
    'subjects_4000-5999_72_imgs', 
    'subjects_6000-7999_72_imgs',
    'subjects_8000-9999_72_imgs']

def get_lightness(path):
    img = cv2.imread(path)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    return hls[:, :, 1]

def avg_lightness_of_person(dir, num=None):
    avg = np.zeros((112, 112))
    
    files = os.listdir(dir)
    print(dir)

    for index in range(len(files)):
        avg += get_lightness(os.path.join(dir, files[index]))
    
    if num is not None:
        random_file = files[random.randint(0, len(files) - 1)]
        img = cv2.imread(os.path.join(dir, random_file))
        cv2.imwrite(f'{num}.png', img)

    return (avg / len(os.listdir(dir))).astype(np.uint8)

def overall_lightness(large_dir):
    overal_avg = np.zeros((112, 112))

    dirs = os.listdir(os.path.join(r'C:\Data', large_dir))
    bar = tqdm(range(len(dirs)))
    bar.set_description(large_dir)

    for index in bar:
        overal_avg += avg_lightness_of_person(os.path.join(r'C:\Data', large_dir, dirs[index]))

    return overal_avg / len(dirs)

# dir1 = r"C:\Data\subjects_2000-3999_72_imgs\2014"
# dir2 = r"C:\Data\subjects_2000-3999_72_imgs\2004"
# dir3 = r"C:\Data\subjects_2000-3999_72_imgs\2017"
# dir4 = r"C:\Data\subjects_2000-3999_72_imgs\2030"
# dir5 = r"C:\Data\subjects_6000-7999_72_imgs\6008"
# dir6 = r"C:\Data\subjects_6000-7999_72_imgs\6024"
# dir7 = r"C:\Data\subjects_6000-7999_72_imgs\6071"

# cv2.imwrite('dir1.png', avg_lightness_of_person(dir1, 1))
# cv2.imwrite('dir2.png', avg_lightness_of_person(dir2, 2))
# cv2.imwrite('dir3.png', avg_lightness_of_person(dir3, 3))
# cv2.imwrite('dir4.png', avg_lightness_of_person(dir4, 4))
# cv2.imwrite('dir5.png', avg_lightness_of_person(dir5, 5))
# cv2.imwrite('dir6.png', avg_lightness_of_person(dir6, 6))
# cv2.imwrite('dir7.png', avg_lightness_of_person(dir7, 7))

for dir in folders:
    avg = overall_lightness(dir)
    cv2.imwrite(fr'overall\{dir}.png', avg)