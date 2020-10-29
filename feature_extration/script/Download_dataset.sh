ln -s ../script/Download_data.py ./script/Download_data.py
ln -s ../data Data
python script/Download_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data
rm -rf Data/hico_20160224_det.tar.gz

python script/Download_data.py 11SLejGxS4psGlOaWL0jc89RP6yVdtq_X Weights.zip
unzip Weights.zip
rm -rf Weights.zip
