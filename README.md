Specificity
===========

Run download_database.m to download the related data

Description of scripts
----------------------

#### Correlations

* _correlations/correlate_attributes.m:_ Correlate specificity with attributes
* _correlations/correlate_color.m:_ Correlate specificity with color
* _correlations/correlate_scores.m:_ Various correlations: specificity with memorability, specificity with other features and inter-human agreement

#### Computing features

* _image_search/features/find_gist.m:_ Find gist features
* _image_search/features/find_objectness.m:_ Find objectness features
* _image_search/features/find_saliency.m:_ Find saliency features

#### Computing specificity

##### Ground truth
* _image_search/search_sentences_mud.py:_ Compute between-image sentence similarities
* _image_search/search_sentences_parameters.py:_ Compute within-image sentence similarities and similarity between query and reference sentence

##### Predicted
* _image_search/compute_specificity_predicted.m:_ Compute predicted specificity for all images in a cross val fashion
* _image_search/predict_specificity_tryfeatures.m:_ Try different features while varying number of training images to see which feature does best

#### Search

* _image_search/search_images_50sentences.m:_ Vary number of sentences and search on ground truth specificity
* _image_search/plot_retrieval_curve.m:_ Plot results from previous script
* _image_search/predict_search_vary_numtrainingimages.m:_ Vary number of training and search on 20% of dataset
* _image_search/plot_predict_search_numtrainingimages.m:_ Plot results from previous script
* _image_search/search_vary_numtestimages.m:_ Vary number of images ranked and search (ground truth and predicted)
* _image_search/plot_vary_numtestimages.m:_ Plot results from previous script

#### Auxiliary functions/scripts

* _aux_functions/progressbar.m:_ Show a progressbar (with dots) while iterating over loops
* _image_search/grid_search.m:_ Do a grid search on SVR parameters C/gamma and return the best values
* _image_search/similary.py:_ Functions to compute tfidf score, find best match and compute sentence similarity
* _image_search/load_search_parameters:_ Load the data produced in image_search/search_sentences_mud.py and image_search/search_sentences_parameters.py

## Citation

Please cite the following publication if you use this code in your research:

@inproceedings{jas2015specificity,
Author = {Mainak Jas and Devi Parikh},
Title = {{Image Specificity}},
Year = {2015},
booktitle = {{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}}
}
