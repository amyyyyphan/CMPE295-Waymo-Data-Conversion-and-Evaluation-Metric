# Instructions for converting Waymo dataset into KITTI format

## References
The instructions are based on these links:

* [Prerequisites](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/get_started.md): Provides the instructions on how to prepare the conda environment

* [Dataset Preparation](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/user_guides/dataset_prepare.md): Provides the instructions on how to convert the Waymo dataset into KITTI format

* [Hyejun Lee's repository](https://github.com/leehj825/CMPE295_mmdetection3d)

* [College of Engineering HPC](https://www.sjsu.edu/cmpe/resources/hpc.php)


## Set up conda environment
1. Create a conda environment
```
conda create -n open-mmlab python=3.8 -y
```

2. Activate conda environment
```
conda activate open-mmlab
```

3. Install PyTorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install CUDA toolkit 11.8
```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

5. Install MMEngine, MMCV, and MMDetection using MIM
```
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
```

6. Install mmdet3d using MIM
```
mim install "mmdet3d>=1.1.0"
```

7. Install waymo-open-dataset library
```
pip install waymo-open-dataset-tf-2-6-0
```


## Download Waymo open dataset v1.4.1
Dataset Link: [Waymo Open Dataset](https://waymo.com/open/download/)

Data Split Link: [Data Split Files](https://waymo.com/open/download/)

1. Create a new folder under the `data` directory and name it `waymo`. The `waymo` folder should look like:
```
mmdetection3d
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── ImageSets
│   │   │   ├── testing
│   │   │   ├── testing_3d_camera_only_detection
│   │   │   ├── training
│   │   │   ├── validation
```

You can download the dataset in HPC using the [gsutil](https://cloud.google.com/storage/docs/gsutil_install). Follow the instructions in the link to install it. Log into the account that has access to the Waymo dataset. Download Waymo open dataset v1.4.1 [here](https://waymo.com/open/download/). When you click on the `Download` button to download multiple `.tfrecord` files, it will give you the gsutil command that you can use to download the files in HPC.

Download the `.tfrecord` files into the corresponding folders in `data/waymo/waymo_format/`

2. Download the data split `.txt` files from [here](https://waymo.com/open/download/) and put it into `data/waymo/kitti_format/ImageSets`


## Convert the Waymo dataset into KITTI format
1. Add these lines to create_data.py towards the beginning of the file:
```
import sys
sys.path.append('./')

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```


### For converting a small amount of data:
1. Enable GPU
```
srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --pty /bin/bash
```

2. Execute create_data.py
```
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --extra-tag waymo
```

You may run into the following errors. Install the necessary libraries and correct versions.

1. **Error**:
    ```
    OSError: /lib64/libc.so.6: version `GLIBC_2.18' not found (required by /home/012194572/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/open3d/libc++abi.so.1)
    ```

    **Solution**:
    ```
    pip install open3d-python
    ```

2. **Error**:
    ```
    ImportError: Please run "pip install waymo-open-dataset-tf-2-6-0" >1.4.5 to install the official devkit first.
    ```

    **Solution**:
    ```
    pip install waymo-open-dataset-tf-2-6-0
    ```

3. **Error**:
    ```
    ImportError: Numba needs NumPy 1.22 or greater. Got NumPy 1.19.
    ```

    **Solution**:
    ```
    pip install numpy==1.22
    ```

4. **Error**:
    ```
    TypeError: Descriptors cannot be created directly.
    If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
    If you cannot immediately regenerate your protos, some other possible workarounds are:
     1. Downgrade the protobuf package to 3.20.x or lower.
     2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

    More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
    ```

    **Solution**:
    ```
    pip install protobuf==3.20.1
    ```


### For converting a large amount of data
Make sure that the required libraries and versions are installed before using this (look at the section for converting a small amount of data).

1. Create a file called conversion.sh in the mmdetection3d directory. Copy this into the file, replacing <email> with your email and <SJSU ID> with your student ID. You may have to make additional changes if you use a different CUDA version or a different name for your conda environment.
```
#!/bin/bash
#
#SBATCH --job-name=dtcv
#SBATCH --output=dtcv-srun.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END

module load cuda/11.8

source /home/<SJSU ID>/anaconda3/etc/profile.d/conda.sh
conda activate open-mmlab

python ./tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --extra-tag waymo

conda deactivate
```

2. Schedule the job
```
sbatch conversion.sh
```

The system will email you when the job is finished. The logs will be stored in `dtcv-srun.log`


### Resulting `kitti_format` directory
After the Waymo dataset is converted into KITTI format, there will be a new directory under `mmdetection3d/data/waymo` called `kitti_format`. It should look like this:
```
mmdetection3d
├── data
│   ├── waymo
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   │   │   ├── testing
│   │   │   ├── testing_3d_camera_only_detection
│   │   │   ├── training
│   │   │   ├── waymo_gt_database
│   │   │   ├── waymo_dbinfos_train.pkl
│   │   │   ├── waymo_infos_test.pkl
│   │   │   ├── waymo_infos_train.pkl
│   │   │   ├── waymo_infos_trainval.pkl
│   │   │   ├── waymo_infos_val.pkl
│   │   ├── waymo_format
```