# Spatially Regularized Parametric Map Reconstruction for Fast Magnetic Resonance Fingerprinting
This repository contains code for the [Medical Image Analysis](https://www.journals.elsevier.com/medical-image-analysis) paper "Spatially Regularized Parametric Map Reconstruction for Fast Magnetic Resonance Fingerprinting", which can be found at https://doi.org/10.1016/j.media.2020.101741.

## Installation

The installation has been tested with Ubuntu 18.04, Python 3.6, TensorFlow 1.15, and CUDA 10.0 (cuDNN 7.6.5). The ``requirements.txt`` file lists all other dependencies.

First, create a virtual environment named `mrf` with Python 3.6:

        $ virtualenv --python=python3.6 mrf
        $ source ./mrf/bin/activate

Second, copy the code:

        $ git clone https://github.com/fabianbalsiger/mrf-reconstruction-media2020
        $ cd mrf-reconstruction-media2020

Third, install the required libraries:

        $ pip install -r requirements.txt

This will install all required dependencies. Please refer to the official TensorFlow documentation on how to use TensorFlow with [GPU support](https://www.tensorflow.org/install/gpu). Note that this repository uses TensorFlow 1.15 instead of 1.10 as mentioned in the paper to be able to use CUDA with version higher than 9.0 (this, however, comes with some deprecation warning, which can be ignored).

## Usage

We shortly describe the training and testing procedure.
Unfortunately, the data used in the paper is not publicly available due to patient confidentiality. But, we provide one image slice of one patient for reference. From this image slice, a script generates dummy data (subjects), which mimics the real data, such that you are able to run the code.

### Dummy Data Generation and Configuration

We handle the data using [pymia](https://pymia.readthedocs.io/en/latest). Therefore, we need to create a hierarchical data format (HDF) file to have easy and fast access to our data during the training and testing.
Create the dummy data by

        $ python ./snippets/create_dataset.py

This will create the file ``./data/data.h5``, or simply our dataset. Use any open source HDF viewer to inspect the file (e.g., [HDFView](https://www.hdfgroup.org/downloads/hdfview/)).
Please refer to the [pymia documentation](https://pymia.readthedocs.io/en/latest/examples.html) on how to create your own dataset. 

We now create so called indices files, which allow fast patch-wise data access. Execute

        $ python ./snippets/create_indices_files.py

This will create a JSON file per subject in the directory ``./data/indices``. Each index basically references a 32 x 32 patch in the images, which we can access through array slicing.

Further, we need to create a training/validation/testing split file by

        $ python ./snippets/create_split.py

This will create the file ``./data/split1_04-02-02.json``.

Finally, we create a region of interest (ROI) reference file (``./data/roi.csv``) for easy evaluation by

        $ python ./snippets/create_roi_reference_file.py

Note that you need to adapt these scripts when using your own data.

### Training
To train the model, simply execute ``./bin/main.py``. The data and training parameters are provided by the ``./bin/config.json``, which you can adapt to your needs.
Note that you might want to specify the CUDA device by

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/main.py

The script will automatically use the training subjects defined in ``./data/split1_04-02-02.json`` and evaluate the model's performance after every 5th epoch on the validation subjects.
The validation will be saved under the path ``result_dir`` specified in the configuration file ``./bin/config.json``.
The trained model will be saved under the path ``model_dir`` specified in the configuration file ``./bin/config.json``.
Further, the script logs the training and validation progress for visualization using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
Start the TensorBoard to observe the training:

        $ tensorboard --logdir=<path to the model_dir>

### Testing
The training script (``./bin/main.py``) directly provides you with the reconstructions of the validation subjects.
So, you need to execute the testing script (``./bin/test.py``) only when you want apply it to your testing set.

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/test.py --model_dir=<patch to your model directory> --result_dir="<path to your result directory>"

#### Permutation Experiments
To perform the permutation experiments, use

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/test.py --model_dir=<patch to your model directory> --result_dir="<path to your result directory>" --do_permutation=True

Note that this might take a while to run.

## Support
We leave an explanation of the code as exercise ;-). But if you found a bug or have a specific question, please open an issue or a pull request.

Generally, adaptions to your MRF sequence should be straight forward. Once you were able to generate the dataset (HDF file), the indices files, the split file, and the ROI reference file, the code should run by itself without massive modifications. 

## Citation

If you use this work, please cite

```
Balsiger, F., Jungo, A., Scheidegger, O., Carlier, P. G., Reyes, M., & Marty, B. (2020). Spatially Regularized Parametric Map Reconstruction for Fast Magnetic Resonance Fingerprinting. Medical Image Analysis, 64, 101741. https://doi.org/10.1016/j.media.2020.101741
```

```
@article{Balsiger2020a,
author = {Balsiger, Fabian and Jungo, Alain and Scheidegger, Olivier and Carlier, Pierre G. and Reyes, Mauricio and Marty, Benjamin},
doi = {10.1016/j.media.2020.101741},
journal = {Medical Image Analysis},
pages = {101741},
title = {{Spatially Regularized Parametric Map Reconstruction for Fast Magnetic Resonance Fingerprinting}},
volume = {64},
year = {2020}
}
```

## License

The code is published under the [MIT License](https://github.com/fabianbalsiger/mrf-reconstruction-media2020/blob/master/LICENSE).