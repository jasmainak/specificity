clear all;

dataset = input('Please enter dataset(pascal/clipart/memorability): ', 's');
files = dir(['../../../data/images/' dataset '/']);

cd('../../../library/objectness-release-v2.2/objectness-release-v2.2/');
startup;

for i=3:length(files)

    image_name = files(i).name;

    fprintf('\nComputing heatmap for image ... %s', image_name);

    filename = ['~/Desktop/projects/specificity/data/image_features/objectness/' ...
                dataset '/' image_name '_objmap.mat'];

    if exist(filename, 'file')
        continue;
    end

    I = imread(['~/Desktop/projects/specificity/data/images/' dataset '/' image_name]);

    windows = runObjectness(I,1000);
    obj_heatmap = rgb2gray(computeObjectnessHeatMap(I,windows));
    set(gcf,'visible','off')

    save(filename, 'obj_heatmap');

    fprintf(' [Done]');
end