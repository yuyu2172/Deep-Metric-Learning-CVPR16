function dict_dist = find_dups


%% generate splits
[image_ids, class_ids, super_class_ids, path_list] = ...
    textread('/cvgl/group/Ebay_Dataset/Ebay_info.txt', '%d %d %d %s', ...
    'headerlines', 1);

% 1. Do one scan and find out how many classes each super class has
dict_dist = containers.Map;

for i = 1:length(image_ids)
    if mod(i,1000)==0
        fprintf('%d/%d\n', i, length(image_ids));
    end
    this_filename = path_list{i};
    this_image_idx  = image_ids(i);
    this_class_idx = class_ids(i);
    this_super_class_idx = super_class_ids(i);
    
    C = strsplit(this_filename, '/');
    this_name = C{2};
    
    if isKey(dict_dist, this_name)
        fprintf('[dup] %s %s\n', this_filename, dict_dist(this_name));
    else
        dict_dist(this_name) = this_filename;
    end
end
    
    

% train_images = {};
% val_images = {};
% dict = containers.Map('keytype', 'double', 'valuetype', 'any');
% 
% for i = 1:length(image_ids)
%     this_filename = path_list{i};
%     this_image_idx  = image_ids{i};
%     this_class_idx = class_ids{i};
%     this_super_class_idx = super_class_ids{i};
%     
%     fprintf('%d/%d, classid=%d, filename= %s\n', i, length(image_ids), ...
%         this_class_idx, this_filename);
%     
%     
%     