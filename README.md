# Attention Free Revolution
Maybe Attention isn't what all we need? This was something that I thought to myself when I learnt in detail about Attention Mechanism and Transformers in general. I know that Attention is the backbone of most problems we solve now in NLP, but what if it was not there? Can we actually replace attention ? Or is it that too much of our SOTA results are due to Attention mechanism, or Transformer.

I love Transformers, but, like my mother says, we really understand someone's importance in our life when they are not there. So, being a lover for such questions, and with a inquisitive mind, started my journey to replace Attention with a different algorithm, and to make an architecture similar to Transformers - which I call Leviathan. 

![maybe_attention](https://github.com/yash-srivastava19/attention-free-revolution/assets/85068689/f2b8cae0-0c0c-48c9-8c3d-f3f7b2135b40)


## What replaces Attention? 
There is not any particular reason why Attention should be not used? Those peeps definitely researched a lot on this, and decided to move forward with the Attention we use today, but I thought of this in a different way which I would like to explain now. 

I read somewhere that self-attention can be seen as a Graph Neural Network, where each token in the input sequence is a node, and edges denote the relationship between each tokens, and that attention layers is a directed graph - which makes sense as different context give different meaning to how different tokens are connected. I remebered that in dot-product attention, we multiplied the **Q** with **K** using dot product - which effectively tells the scores of similarity between **Q** and **K** . Just for remider : 

<img width="700" alt="4mhWz" src="https://github.com/yash-srivastava19/attention-free-revolution/assets/85068689/7095f539-f5ec-4fd4-b070-05a2bda31c6a">

For starters, I started thinking of tokens as signals, and self-attention as a measure of correlation between those signals. Attention can be used to capture both long range dependencies, and delay, whereas correlation is great when there is delay in signals. Cross-correlation is a more general way to find similarity between signals(tokens) effectively, dot product is just cross-correlation with zero lag. Introduction of lags explicitly allows for greater in-context relationships(hypothesis). With this in mind, I started testing of methods I needed to mimic attention - using signal correlations. 

Now, back to that graph analogy. Suppose instead of vanilla dot product, we used cross correlation - which uses sliding window product(similar to convolution), and now due to the 'lag' in the tokens(signal), there are effectively more nodes in the graph - which allows for more ways in which tokens can be connected. This, as far as I see, allows to give more context, as now we can have more context (more ways) and different ways in which a bunch of tokens interact. From the definition, we have : 

> Cross correlation is a measure of two series as a function of the displacement of one relative to the other. It is similar to convolution of two functions, and can be used to measure the degree between data.

Convolutions have not been used in Transformer type architectures, despite the fact that they capture positional information is because Transformers provide flexibility. We can obtain learnable embeddings in Transformers, but not from CNNs - as the kernel become static once learned(also, computationally expensive). 

With Attention, it is different. Query, Key, Value matricies allow context to be taken into account.

Involution is between those two. It is more effective and efficient than CNNs, and are much simpler that self-attention.

I have the plan to introduce some kind of involution score as an alternative to attention, but that is a different topic, and I need to discuss a lot with other. I tested a lot with these. Even with the smallest of things.

## What replaces Transformers?
Leviathan is my replacement of Transformers. I was reading about different architectures of models such as BERT, BART, GPT-1, GPT-2, GPT-3, GPT-4... Wait a second. We dont' know much about GPT-4, but from comments by Geofry Holtz, and others, it's a combination of several models. Now I know a thing or two about model-ensembling and joint training, so with the power God and PyTorch on my side, I came up with a family of architectures, or as I like to call them - Flavours of Leviathan, each of which contain a bunch of model that are jointly trained with an cross-correlation interaction.

Although not tested due to computational power limitations, I propose a LeviathanModelEnsemble(Giant Leviathan) which consists of 8 model in ensemble, which are divided in 4 modules of 2 models each. Where, I suppose we can have every module(2 models) can be trained on a different data(or different type of data), and then we can combine the results of all the modules using only the relevant part(from MultiHeadedCorrelation). 

Although the exact details are different types of models are yet to be finalised, I will have different flavours of Leviathan, all of which build upon the `LeviathanComponentBase`, and I recommend this as a great exercise to other people also, who want to make their own models. It's a great learning exercise as well. PRs with an architeure build using these components will make it to the official repository.      

## Experimentation Details
Model experimentation will be done once we have idea of how to train these huge models(and collection of models). The baseline `LeviathanComponent` is a 20 headed, 20 blocks, 800 embedding dimension model(around 77M Parameter)

## Can you actually replace Attention?
Although I tested with very few data, from the initial results, I can only say that the attention score and cross-correlation scores are *similar*. Deciding on what to train the data on, and what architecture do we train needs further discussion with people. Till now, we can enjoy attention, and look scepticaly at correlation. 

## Future Plans?
More and more and more models using this architecture. I need to make a family of these models using correlation, and test it against the standard attention models. What if it's a start of something new?

## Was it worth it?
Yes. Every single ounce of it was worth it.
