function gen_caffe_dataset_multilabel_m128(mode, batch_size)

if nargin == 0
    mode = 'train';
    batch_size = 128*1;
end 

conf = config;

%% Load images 
if strcmp(mode, 'train')
    images = load(fullfile(conf.root_path, ...
        'training_images_crop15_square256.mat'), 'training_images');
    images = images.training_images;
    assert(length(images)==59551);
elseif strcmp(mode, 'val')
    images = load(fullfile(conf.root_path, ...
        'validation_images_crop15_square256.mat'), 'validation_images');
    images = images.validation_images;
else
    error('unknown mode %s', mode);
end

%% Pick image pairs
assert(mod(batch_size, 2)==0);
[image_id_pairs, labels] = get_training_examples_multilabel(mode, batch_size);

%% Prep dataset for C and Caffe conventions
images = cat(4, images.img); 

assert(max(image_id_pairs(:)) <= length(images));
% images must be (height x width x channels x num)
assert(ndims(images) == 4); 
% assume square images
assert(size(images, 1) == size(images, 2));
% images must have 3 channels
assert(size(images, 3) == 3);

% Convert to BGR
images = images(:,:,[3 2 1],:);
% Switch width and height
images = permute(images, [2 1 3 4]);

% Caffe likes 0-1 labels
labels = int32(labels);
labels(labels == -1) = 0;

labels_cont = 1:11318;
assert(isequal(double(unique(labels)), labels_cont'));

% Subtract 1 for C indexing
image_id_pairs = int32(image_id_pairs - 1);

% check the number of paired data is divisible 

%% Do a morph before the mex file

image_id_pairs_serial = int32(zeros(numel(image_id_pairs), 1));
labels_serial = int32(zeros(length(labels)*2, 1));
 
insert_idx = 1;
lookup_idx = 1;
for i = 1:length(labels)
    if mod(i-1, (batch_size/2)) == 0
        image_id_pairs_serial(insert_idx : insert_idx + batch_size-1, :) = ...
            reshape(image_id_pairs(lookup_idx : lookup_idx + batch_size/2-1,:), [], 1);
        
        labels_serial(insert_idx : insert_idx + batch_size-1) = ...
            reshape(labels(lookup_idx : lookup_idx + batch_size/2 -1, :), [], 1);
        
        insert_idx = insert_idx + batch_size;
        lookup_idx = lookup_idx + batch_size/2;
    end
end
assert(length(image_id_pairs_serial) == 2*length(labels));
assert(mod(length(image_id_pairs_serial), batch_size)==0);

%% Write to lmdb
fprintf('Writing level db..\n');
if strcmp(mode, 'train')
    filename = conf.training_set_path_multilabel_m128;
elseif strcmp(mode, 'val')
    filename = conf.validation_set_path_multilabel;
else
    error('unknown mode %s', mode);
end

serialized_pairs_to_leveldb(images, image_id_pairs_serial, labels_serial, filename); 
fprintf('Done writing level db..\n');  
