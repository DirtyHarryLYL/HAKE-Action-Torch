mkdir data
python script/Download_data.py 1c5obmpgtybcRLh-EizGCcOouOqrZPRnN data/db_trainval_with_pool.pkl
python script/Download_data.py 1bXAC4SyAW2G4WtEAsoMulOCGIKfHoTj8 data/cand_positives_trainval.pkl
python script/Download_data.py 1mgY8DsTSWouRSkCnaF6gdMdZ_QrL4akS data/cand_negatives_trainval.pkl
python script/Download_data.py 1pZk-OC31k5P_RIDR0KpDFBC-4Xt9dBdc data/db_test_feat.pkl
python script/Download_data.py 1blaR80wqQaelIeEomfq3VtGIAzRNRvDA data/candidates_test.pkl

mkdir data_drg
python script/Download_data.py 1Z8WMEmg8T81kveksB3jm28j3OBs1cGjs data_drg/db_test_feat.pkl
python script/Download_data.py 1GSddLTozPobzMTgxJZNsdFchur9qdodG data_drg/candidates_test.pkl

mkdir data_vcl
python script/Download_data.py 1DGX7fCFbbxGIa5E8p5GlDxmstTgUD3T5 data_vcl/db_test_feat.pkl
python script/Download_data.py 12U-QI5ixtDPduRh2VHmV-7Dy8PvFOGZz data_vcl/candidates_test.pkl


mkdir data/Union_feat/
python script/Download_data.py 1UnXos3fbpmgl9ONVzgChjHiU1YuG7Cyc data/Union_feat/test.7z.001
python script/Download_data.py 1ecvF6dOQCM2TIw66czUiD3na46_FPS8c data/Union_feat/test.7z.002
python script/Download_data.py 1Y115lzCgqjRc5kMPfJQDjzkubnlAF4K9 data/Union_feat/test.7z.003
python script/Download_data.py 1EF7bICo-H1Iy7-J03MqwvN5z-iH12z3A data/Union_feat/test.7z.004
python script/Download_data.py 16bRG-w18hv54_n1utumlsoxcbGoF4rI9 data/Union_feat/test.7z.005
python script/Download_data.py 1F5SyAwZxSEAZVZ8UXryklLdaeKyR0Z5T data/Union_feat/test.7z.006

mkdir data/feature/
python script/Download_data.py 1ZqFi6v-3umwdPrilyZ9JxH5JjyOQvkkN data/feature/test.7z.001
python script/Download_data.py 1b8o561vNe0u1LZjRmVAC7XIlu9ttMUwH data/feature/test.7z.002

apt install p7zip-full
7z x data/Union_feat/test.7z.001 -r -odata/Union_feat/
7z x data/feature/test.7z.001 -r -odata/feature/
rm -rf data/Union_feat/test.7z.*
rm -rf data/feature/test.7z.*

mkdir exp
python script/Download_data.py 1QZuDE7F0SqFvzO0ZJCZxoszE-qGyuMU_ exp/nis.pkl
mkdir exp/IDN_IPT_hico/
python script/Download_data.py 15Ev-O4dyghxHw_O2jjri1kvyato5Mr8I exp/IDN_IPT_hico/epoch_30.pth

unzip gt_hoi_py2.zip
