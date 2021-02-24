import sys
import download_images
import time
import os

def main(class_csv_path):
    for line in open(class_csv_path):
        print('Downloading {}...'.format(line.strip()))
        download_images.main(os.path.join('metadata/', \
                             line.strip().lower().replace(' ', '_') + '.csv'),\
                             'images/')
        time.sleep(60)

if __name__ == '__main__':
    main(sys.argv[1])
