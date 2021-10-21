## Steps of Installing Activity2Vec
 ### Prerequisites
 - Python 3.7
 - CUDA 10.0
 - Anaconda 3(optional)
 
 ### 1. Create a new conda environment(optional)
 ```
 conda create -y -n activity2vec python=3.7
 conda activate activity2vec
 conda install pip
 ```
 ### 2. Install the dependencies
 ```
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
 git clone https://github.com/DirtyHarryLYL/HAKE-Action-Torch
 cd HAKE-Action-Torch && git checkout Activity2Vec
 pip install -r requirements.txt
 pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
 sudo apt install -y libturbojpeg libturbojpeg-dev
 ```

 ### 3. Setup AlphaPose and Activity2Vec
 ```
 cd AlphaPose && python setup.py build develop && cd ..
 python setup.py build develop
 ```
