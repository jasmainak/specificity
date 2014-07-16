function find_gist()

addpath(genpath('../../library/gistdescriptor/'));
calculate_gist('memorability');
calculate_gist('pascal');

end

function calculate_gist(dataset)

files = dir(['../../data/images/' dataset '/']);

param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

for i=3:length(files)

    image_name = files(i).name;

    fprintf('\nComputing gist descriptor for image ... %s', image_name);

    filename = ['~/Desktop/projects/specificity/data/image_features/gist/' ...
        dataset '/' image_name '_gist.mat'];

    if exist(filename, 'file')
        continue;
    end

    img_file = ['~/Desktop/projects/specificity/data/images/' dataset '/' image_name];
    I = imread(img_file);

    gist_features = LMgist(I, '', param);

    save(filename, 'gist_features');

    fprintf('[Done]');
end

end