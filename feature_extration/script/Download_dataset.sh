ln -s ../script/Download_data.py ./script/Download_data.py
mkdir Data
python script/Download_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data
rm Data/hico_20160224_det.tar.gz

mkdir Weights