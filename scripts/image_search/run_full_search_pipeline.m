% RUN_FULL_SEARCH_PIPELINE runs the entire pipeline to do image search
%
% AUTHOR: Mainak Jas

datasets = {'pascal', 'clipart'}; overwrite = 1;
n_jobs = 6; % number of parallel processes to run for computing similarity scores

for i=1:length(datasets)
    dataset = datasets{i};

    % Calculate similarity scores in python
    system(sprintf('python search_sentences_parameters.py --dataset %s -j %d -t train_pos_class', dataset, n_jobs));
    system(sprintf('python search_sentences_parameters.py --dataset %s -j %d -t train_neg_class', dataset, n_jobs));
    system(sprintf('python search_sentences_parameters.py --dataset %s -j %d -t test', dataset, n_jobs));

    % Calculate subset of similarity scores out of candidate similarity scores
    % which will be used for calculation of ground-truth specificity
    calculate_train_similarity_indices(dataset);

    % Save ground-truth specificity for training
    save_train_specificity(dataset, overwrite);

    % Calculate predicted LR parameters
    calculate_predicted_LR_specificity(dataset);

end

% Calculate the curve for percentage
calculate_percentage_curve();

% Plot the results
system('python plot_percentage_curve.py');