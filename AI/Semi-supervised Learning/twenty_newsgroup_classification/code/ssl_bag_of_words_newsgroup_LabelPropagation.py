### Now we have shown using t-SNE that the 4-ng data set exhibits clustering, let's try
### and use scikit-learns LabelPropagation algorithm.

'''
We have 500 labelled instances, and using an SVM on the 500 achieves x% on the test of the data.

Label the unlabelled instances -1, and feed into the LP algorithm (see my github example).
Check whether we now have a better accuracy on the test data than the supervised SVM.

Another thing we could do is use the results from LabelPropagation that the model is sure about as pseudo-labels
to the SVM... (like we did in the NN case).

We could also use the results of LabelPropagation and LDA as additional inputs for the SVM.

Test all of these things.


'''