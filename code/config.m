function conf = config()

conf = struct();

%% Directories


conf.root_path = '/home/leus/playground/Deep-Metric-Learning-CVPR16/';
conf.image_path = '/home/leus/.chainer/dataset/pfnet/chainercv/online_products/Stanford_Online_Products/';
conf.cache_path = '/home/leus/playground/Stanford_Online_Products/';
% conf.root_path = '/scail/scratch/group/cvgl/hsong/Deep-Lifting-for-Metric-Learning-CVPR/code/ebay/';
% conf.cache_path = '/scail/scratch/group/cvgl/hsong/Deep-Lifting-for-Metric-Learning-CVPR/code/ebay/cache';
% conf.image_path = '/cvgl/group/Ebay_Dataset/';

%% Training parameters
conf.preprocessing.crop_padding = 15;
conf.preprocessing.square_size = 256;
conf.preprocessing.num_to_load = 255;
conf.preprocessed_image_file = [conf.cache_path, '/training_images.mat'];
            
% path_triplet = '/cvgl/u/hsong/ebay/cache';
path_triplet = '/home/leus/playground/Stanford_Online_Products_triplet/';


% for multilabel pairs batchsize = 128
conf.training_set_path_multilabel_m128 = [path_triplet, '/training_set_cars196_multilabel_m128.lmdb'];

% for multilabel pairs batchsize = 128*2 = 256
conf.training_set_path_multilabel_m256 = [path_triplet, '/training_set_cars196_multilabel_m256.lmdb'];

% for multilabel pairs batchsize = 128*3 = 384
conf.training_set_path_multilabel_m384 = [path_triplet, '/training_set_cars196_multilabel_m384.lmdb'];

% for debuggin,
conf.training_imageset_path = [path_triplet, '/training_imageset_cars196.lmdb'];

conf.training_set_path_triplet = [path_triplet, '/training_set_triplet.lmdb'];
conf.validation_set_path_triplet = [path_triplet, '/validation_set_triplet.lmdb'];
conf.validation_set_path = [path_triplet, '/validation_set_cars196.lmdb'];
