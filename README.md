# The Library of Efficient Transformers for Sequential Recommendation

Code for DASFAA-2023 submission:
> Towards Efficient and Effective Transformers for Sequential Recommendation  
(*Running Title*: Towards Efficient Transformers for Sequential Recommendation)

*Note*: this library is being updated continuously.

The library is built upon PyTorch and RecBole for reproducing recommendation algorithms based on Transformers and then exploring their effectiveness and efficiency.


## Requirements

```
pytorch>=1.7.0
python>=3.7.0
recbole>=1.0.0
```

## Implemented Models

The implemented models can be seen in *the library of efficient Transformers*.

path: /recbole/model/efficient_transformer_recommender/*   
/recbole/model/transformer_layers.py  
/recbole/properties/model/*


*e.g.*, Linformer (path: /recbole/model/efficient_transformer_recommender/linformer.py),  
Performer (path: /recbole/model/efficient_transformer_recommender/performer.py),  
Synthesizer (path: /recbole/model/efficient_transformer_recommender/synthesizer.py), etc.
