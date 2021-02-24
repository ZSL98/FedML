import pandas
import sys
import csv
import os
import requests
from requests.exceptions import HTTPError
import time
from multiprocessing import Pool
import numpy as np

IMG_PER_DIR = 1000
MULTI_PROC = 4
class_img_path = ''

def download(img_id, url):
    dir_seq = int((img_id - 1) / IMG_PER_DIR)
    img_path = os.path.join(class_img_path, '{:03d}'.format(dir_seq))
    if not os.path.exists(img_path):
        os.makedirs(img_path)
        # delay for a while so we won't access Flickr too fast
        # time.sleep(30)
    # retry several times
    success = False
    for i in range(3):
        try:
            r = requests.get(url)
            r.raise_for_status()
        except:
            time.sleep(5)
            continue
        else:
            success = True
            break

    if not success:
        print('Fail when downloading image {}..'.format(img_id))
        return

    with open(os.path.join(img_path, '{:05d}.jpg'.format(img_id)), 'wb') as f:
        f.write(r.content)
        
    return

def worker(img_df):
    img_df.apply(lambda row: \
                 download(row['img_id'], row['url']),\
                 axis=1)
    return

def main(csv_file_path, img_path):
    global class_img_path
    # Create dirs
    path, fn = os.path.split(os.path.splitext(csv_file_path)[0])
    class_img_path = os.path.join(img_path, fn)
    if not os.path.exists(class_img_path):
        os.makedirs(class_img_path)

    # Load CSV
    img_df = pandas.read_csv(csv_file_path, delimiter=',', quotechar='|', \
                             quoting=csv.QUOTE_MINIMAL, 
                             usecols=('img_id', 'url'))
#header=None, \
#names=('img_id', 'lat', 'lon', 'lic', 'acc', 'url'),
                             

    #img_df.apply(lambda row: \
    #             download(row['img_id'], row['url'], class_img_path),\
    #             axis=1)
    pool = Pool(processes = MULTI_PROC)
    img_df_split = np.array_split(img_df, MULTI_PROC)
    pool.map(worker, img_df_split)
    pool.close()
    pool.join()
    return

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
