from utils.similarity import find_sentence_similarity

sent1 = 'Two young men trying to fix their car together'
sent2 = 'The lot at the Toyota dealership is crowded with cars.'

sim = find_sentence_similarity([sent2], [sent1], dataset_name='memorability',
                               method='cosine')
