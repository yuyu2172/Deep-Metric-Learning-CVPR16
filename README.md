# Deep Metric Learning via Lifted Structured Feature Embedding
This repository has the source code and the Stanford Online Products dataset for the paper "Deep Metric Learning via Lifted Structured Feature Embedding" (CVPR16). The paper preprint is available on [arXiv](http://arxiv.org/abs/1511.06452). If you just need the Caffe code, check out the [Submodule](https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16).

## Citing this work
If you find this work useful in your research, please consider citing:

    @inproceedings{songCVPR16,
        Author = {Hyun Oh Song and Yu Xiang and Stefanie Jegelka and Silvio Savarese},
        Title = {Deep Metric Learning via Lifted Structured Feature Embedding},
        Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
        Year = {2016}
    }

## Installation
1. Install prerequsites for `Caffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. Compile the `Caffe-Deep-Metric-Learning-CVPR16` Github submodule.

## Training procedure 
1. Download pretrained GoogLeNet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
2. Generate the LMDB file to convert the training set of images to the DB format. Example scripts are in `code/` directory.
 * Modify and run `code/compile.m` to mex compile the cpp files used for LMDB generation.
 * Modify `code/config.m` to set save paths.
 * Run `code/gen_caffe_dataset_multilabel_m128.m` to start the LMDB generation process.
3. Create the `model/train*.prototxt` and `model/solver*.prototxt` files. Please refer to the included `*.prototxt` files in `model/` directory for examples.
4. Inside the caffe submodule, launch the Caffe training procedure.
`caffe/build/tools/caffe train -solver [path-to-training-prototxt-file] -weights [path-to-pretrained-googlenet] -gpu [gpuid]`

## Feature extraction after training
1. Modify and run `code/gen_caffe_validation_imageset.m` to convert the test images to LMDB format.
1. Modify the test set path in `model/extract_googlenet*.prototxt`.
2. Modify the model and test set path and run `code/compute_googlenet_distance_matrix_cuda_embeddings_liftedstructsim_softmax_pair_m128.py`.

## Clustering and Retrieval evaluation code
Coming soon in two of weeks.

## Stanford Online Products dataset
Coming soon in two of weeks.

## Licence
MIT Licence
