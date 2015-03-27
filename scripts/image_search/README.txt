Run scripts in following order:

1. search_sentences_parameters.py to generate positive class and query similarity scores
2. search_sentences_mud.py to generate negative class similarity scores
3. leave_img_similarities_out.m to generate indices of contaminated sentence pairs (one sentence is from a predicted image / a ref sentence / a query sentence)
4. save_clean_specificity.m to generate the ground-truth specificity parameters
5. search_specificity.m
