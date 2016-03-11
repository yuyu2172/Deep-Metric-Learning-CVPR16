conf = config;

load([conf.root_path, 'splits.mat'], 'train_images', 'val_images');

% check if all images are readible
%check_images(conf.image_path, train_images);

training_images = load_cropped_images(conf.image_path, train_images, ...
    conf.preprocessing.crop_padding, conf.preprocessing.square_size);
    
savefast(fullfile(conf.root_path,...
    'training_images_crop15_square256.mat'), 'training_images');

validation_images = load_cropped_images(conf.image_path, val_images, ...
    conf.preprocessing.crop_padding, conf.preprocessing.square_size);
    
savefast(fullfile(conf.root_path,...
    'validation_images_crop15_square256.mat'), 'validation_images');