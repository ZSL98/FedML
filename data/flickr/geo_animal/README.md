# Geographical Distribution of Mammal Images 

This dataset is created based on the geographical distribution of common mammal images on Flickr. 

In this dataset, we provide the metadata of all images in this dataset, including their geotags, license status, URL to the images, and their corresponding countries and geographical regions (based on [UN M49 Standard](https://unstats.un.org/unsd/methodology/m49/))

The following steps are tested on Ubuntu 16.04 and Ubuntu 18.04 with python 3.6.9. The download scripts should work on most python3 platforms.

## Download the Images

Extract the tarball of the dataset, and switch into the root directory:

```
tar zxvf geo_animal.tar.gz
cd geo_animal
```

All the metadata are in the `metadata` folder (in CSV format). One can refer to `region_names.csv` to find the mapping between region codes and region names.

To begin downloading, install the dependent python3 modules by running the following command:

```
pip3 install -r requirements.txt
```

Download all images by running the following command. Note that this step can take a while so it is strongly recommended to run this command in an environment that won't terminate running programs when the terminal is disconnected (such as [tmux](https://github.com/tmux/tmux)).

```
python3 download_list.py mammal.csv
```

All the downloaded images will be saved in the `images` folder.

In our experience, few downloads may not succeed because Flickr is not reliable all the time. Hence, we provide a script to list all downloaded file. The user can then join the downloaded list with the metadata to get all the information for the successfully downloaded images. For example:

```
python3 list_download_files.py mammal.csv images/ dlfiles/
```
