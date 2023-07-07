# Attention Free Revolution
Maybe Attention isn't what all we need? This was something that I thought to myself when I learnt in detail about Attention Mechanism and Transformers in general. I know that Attention is the backbone of most problems we solve now in NLP, but what if it was not there? Can we actually replace attention ? Or is it that too much of our SOTA results are due to Attention mechanism, or Transformer.

I love Transformers, but, like my mother says, we really understand someone's importance in our life when they are not there. So, being a lover for such questions, and with a inquisitive mind, started my journey to replace Attention with a different algorithm, and to make an architecture similar to Transformers - which I call Leviathan. 

## What replaces Attention? 
There is not any particular reason why Attention should be not used? Those peeps definitely researched a lot on this, and decided to move forward with the Attention we use today, but I thought of this in a different way which I would like to explain now. 

I read somewhere that self-attention can be seen as a Graph Neural Network, where each token in the input sequence is a node, and edges denote the relationship between each tokens, and that attention layers is a directed graph - which makes sense as different context give different meaning to how different tokens are connected. Also, I was reading about convolutions, and then something clicked.  

I started thinking of tokens as signals, and self-attention as a measure of correlation between those signals, and from there it was deliberate testing of methods I needed to mimic attention - using signal correlations. From the definition, we have : 

``` Cross correlation is a measure of two series as a function of the displacement of one relative to the other. It is similar to convolution of two functions, and can be used to measure the degree between data. 
```
Convolutions have not been used in Transformer type architectures, despite the fact that they capture positional information is because Transformers provide flexibility. We can obtain learnable embeddings in Transformers, but not from CNNs - as the kernel become static once learned(also, computationally expensive). 

With Attention, it is different. Query, Key, Value matricies allow context to be taken into account.

Involution is between those two. It is more effective and efficient than CNNs, and are much simpler that self-attention.

I tested a lot with these. Even with the smallest of things.

## What replaces Transformers?
Leviathan is my replacement of Transformers. I was reading about different architectures of models such as BERT, BART, GPT-1, GPT-2, GPT-3, GPT-4... Wait a second. We dont' know much about GPT-4, but from comments by Geofry Holtz, and others, it's a combination of several models. Now I know a thing or two about model-ensembling and joint training, so with the power God and PyTorch on my side, I came up with a family of architectures, or as I like to call them - Flavours of Leviathan, each of which contain a bunch of model that are jointly trained.


## Experimentation Details


## Can you actually replace Attention?


## Future Plans?


## Was it worth it?
Yes. Every single ounce of it was worth it.
