## Introduction


In language modeling self-supervised learning has exploded in the recent years, with the advent of models like GPT-3 and BERT, this was made possible by two features of language:
* Good learning objectives: since we have an intuiton on how language works, we hand-crafted self-supervised learning objectives like predict the next word given the ones before(GPT), or predict whether a sentence is the logical continuation of another(BERT)
* Language provides context information very quickly: we need few words to understand the used language, or even the task that we need to solve, in other words, language models can extract the context by the phrase itself

Now consider tabular data, we can solve the first issue by using expert knowledge or simply training on a big labeled dataset(like in vision)  
What about the second one?  
Can a single training example provide the whole context like in language?  
Not at all.  

For example suppose that we train on the adult dataset which has 14 features and the task is to predict whether a person earns more than 50k a year or not, now we want to apply our pretrained model on another dataset, to make things simple consider that this other dataset is the adult dataset as well but with the columns swapped, obviously the model will fail miserably, this is because having the columns shuffled can happen in tabular data but shuffling the words doesn't happen in language, which has a very clear, universal way of providing information.  

This is a big problem because it means that we can't use this model on other datasets, even if these datasets should be close to the one we pretrained on.  

So the problem is providing context information to our training routine

## Providing context

As we said language models can understand the context by only looking at the example itself (Zero-shot predictions), this is obviously impossible for tabular data: Considering the adult dataset example before the model can't learn all the shuffling that we made from a single example itself because it can happen that two feature values are very close to each other, or maybe we performed standard-scaling on the inputs. So from now on we will focus on supervised pretraining on a large dataset and then fine tuning on the required task  

The general formula to provide context is: pre-analyze a dataset to extract some features, which can either be global(dataset-wide) or local(example only), then use the extracted features in a big pretrained net. While fine-tuning we only need to tune the context part of the model.

## Example models

Suppose we have a very big dataset with 10k features that represent the current life of 1 Billion individuals on the planet and one of the features is whether the person's income is >50k per year. A model pretrained on this data should perform very well on the adult dataset which however has only 14 features.  

One of the way to use this trained model is by using an encoder-decoder architecture: We only train the encoder on the adult dataset, leaving the decoder the same. This paper will explore this approach
