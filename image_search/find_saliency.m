clear all;

dataset = input('Please enter dataset(pascal/clipart/memorability): ', 's');
files = dir(['../../data/images/' dataset '/']);

addpath(genpath('../../library/JuddSaliencyModel/JuddSaliencyModel/'));

for i=3:length(files)
    
    image_name = files(i).name;

    fprintf('\nComputing saliency map for image ... %s\n', image_name);
    
    filename = ['~/Desktop/projects/specificity/data/image_features/saliency/' ...
                dataset '/' image_name '_saliencymap.mat'];
            
    if exist(filename, 'file')
        continue;
    end
    
    img_file = ['~/Desktop/projects/specificity/data/images/' dataset '/' image_name];       
   
    saliencyMap = saliency(img_file);
    set(gcf,'visible','off')
    
    save(filename, 'saliencyMap');
    
    fprintf('\n[Done]');
end