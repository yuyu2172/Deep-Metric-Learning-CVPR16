function gen_caffe_validation_imageset

% only validation images
% don't worry about padding. Use small batch size for p5 feat extraction

conf = config;

images_v = load(fullfile(conf.root_path, ...
    'validation_images_crop15_square256.mat'), 'validation_images');
images_v = images_v.validation_images;

images = cat(4, images_v.img);

validation_filenames = {images_v.filename};

assert(length(images) == length(validation_filenames));

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

savefilename = conf.validation_set_path;
imageset_to_leveldb(images, savefilename);
fprintf('Done writing level db\n');

save(fullfile(conf.cache_path, 'validation_filenames.mat'), 'validation_filenames');
