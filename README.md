# music_genre_classification
Prog Rock Music genre Classification


data_extraction.py: generate the snipped data.
model.py: all the models
training.py: use to train and validate the model.

TODO:
make num_workers=1, if you get any memory error.
specgram needs model changes.


data_extraction.py: Run it to get the snippets
training.py: for training the model
model.py: have all the models.
get_results.py: we use this to generate confusion matrix and accuracy and AUC curve for songs from snippets