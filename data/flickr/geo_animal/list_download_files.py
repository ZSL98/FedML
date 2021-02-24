import pandas as pd
import sys
import csv
import os
import numpy as np
from itertools import repeat

def get_img_id(fn):
    pure_fn = os.path.splitext(os.path.split(fn)[1])[0]
    return int(pure_fn)

def list_files(image_path, result_csv_file, class_id):
    sub_dir = [os.path.join(image_path, d) for d in os.listdir(image_path) \
               if os.path.isdir(os.path.join(image_path, d))]
    result_list = []
    for d in sub_dir:
        files = [os.path.join(d, f) for f in os.listdir(d) \
                 if os.path.isfile(os.path.join(d, f))]
        img_ids = list(map(get_img_id, files))
        result_list.extend(list(zip(img_ids, files, repeat(class_id))))
        
    res_df = pd.DataFrame(result_list, columns = ['img_id', 'path', 'class'])
    res_df.to_csv(path_or_buf = result_csv_file, sep = ',',\
                  quotechar='|', index = False, quoting=csv.QUOTE_MINIMAL)
    return

def main(class_csv_path, image_path, result_csv_path):
    if not os.path.exists(result_csv_path):
        os.makedirs(result_csv_path)

    class_id = 0
    for line in open(class_csv_path):
        print('Processing {}...'.format(line.strip()))
        class_name = line.strip().lower().replace(' ', '_')
        list_files(os.path.join(image_path, class_name),\
                   os.path.join(result_csv_path, class_name + '.csv'),\
                   class_id)
        class_id += 1
    return


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] ,sys.argv[3])
