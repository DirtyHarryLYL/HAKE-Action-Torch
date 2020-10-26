mkdir data
python script/Download_data.py 1c5obmpgtybcRLh-EizGCcOouOqrZPRnN data/db_trainval_with_pool.pkl
python script/Download_data.py 1bXAC4SyAW2G4WtEAsoMulOCGIKfHoTj8 data/cand_positives_trainval.pkl
python script/Download_data.py 1mgY8DsTSWouRSkCnaF6gdMdZ_QrL4akS data/cand_negatives_trainval.pkl
python script/Download_data.py 1pZk-OC31k5P_RIDR0KpDFBC-4Xt9dBdc data/db_test_feat.pkl
python script/Download_data.py 1blaR80wqQaelIeEomfq3VtGIAzRNRvDA data/candidates_test.pkl

mkdir data/Union_feat/
python script/Download_data.py 1UnXos3fbpmgl9ONVzgChjHiU1YuG7Cyc data/Union_feat/test.7z.001
python script/Download_data.py 1ecvF6dOQCM2TIw66czUiD3na46_FPS8c data/Union_feat/test.7z.002
python script/Download_data.py 1Y115lzCgqjRc5kMPfJQDjzkubnlAF4K9 data/Union_feat/test.7z.003
python script/Download_data.py 1EF7bICo-H1Iy7-J03MqwvN5z-iH12z3A data/Union_feat/test.7z.004
python script/Download_data.py 16bRG-w18hv54_n1utumlsoxcbGoF4rI9 data/Union_feat/test.7z.005
python script/Download_data.py 1F5SyAwZxSEAZVZ8UXryklLdaeKyR0Z5T data/Union_feat/test.7z.006
apt install p7zip-full
7z x data/Union_feat/test.7z.001 -r -o data/Union_feat/
rm -rf data/Union_feat/test.7z.*

mkdir exp
python script/Download_data.py 1hLXJKVu9FElD2DdbbuAnCVYI7hxrMPLn exp/nis.pkl
mkdir exp/IDN_IPT_hico/
python script/Download_data.py 19cRqSeRu3Svuc7TA7Zso_uj9F3DwW-zI exp/IDN_IPT_hico/epoch_30.pth
