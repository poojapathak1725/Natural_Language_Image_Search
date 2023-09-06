# Natural Language Image Search

Main objective of our project: Build an image search engine to retrieve images relevant to the input query text


This project aims at query based image retrieval/recommendation. To implement this, we use the dual encoder architecture, where one of the encoder is the image encoder (ResNet50), and the text encoder is the BERT Encoder, which is essentially a transformer. Thus, the images and query text, processed from the COCO Dataset are fed into the dual encoder architecture, where we obtain the embeddings of the text along with the feature vector of the images. These feature vectors and embeddings are passed through the projection embedding layer, where the dimensions of both the encoded outputs are matched and we further evaluate the similarity and the distance, using the Cross Entropy Loss Function and we train our model to learn what kind of textual embeddings correspond to what kind of the encoded feature vectors of the images. Thus, we finally test the model and evaluate our results.

