# The Library of Efficient Transformers for Sequential Recommendation

The library is built upon PyTorch and RecBole for reproducing recommendation algorithms based on Transformers and then exploring their effectiveness and efficiency.


## Requirements

```
pytorch>=1.7.0
python>=3.7.0
recbole>=1.0.0
```

## Model List

| Model Name  | Model Path                                                   | Property Path                             |
| ----------- | ------------------------------------------------------------ | ----------------------------------------- |
| LinearTrm   | recbole/model/efficient_transformer_recommender/lineartrm.py | recbole/properties/model/LinearTrm.yaml   |
| Linformer   | recbole/model/efficient_transformer_recommender/linformer.py | recbole/properties/model/Linformer.yaml   |
| MLPMixer    | recbole/model/efficient_transformer_recommender/mlpmixer.py  | recbole/properties/model/MLPMixer.yaml    |
| Performer   | recbole/model/efficient_transformer_recommender/performer.py | recbole/properties/model/Performer.yaml   |
| Synthesizer | recbole/model/efficient_transformer_recommender/synthesizer.py | recbole/properties/model/Synthesizer.yaml |
| HaloNet     | recbole/model/efficient_transformer_recommender/halonet.py   | recbole/properties/model/                 |

**Please consider to cite our paper if this framework helps you, thanks:**

```
@inproceedings{sun2023towards,
  title={Towards Efficient and Effective Transformers for Sequential Recommendation},
  author={Sun, Wenqi and Liu, Zheng and Fan, Xinyan and Wen, Ji-Rong and Zhao, Wayne Xin},
  booktitle={International Conference on Database Systems for Advanced Applications},
  pages={341--356},
  year={2023}
}
@inproceedings{zhao2022recbole,
  title={RecBole 2.0: towards a more up-to-date recommendation library},
  author={Zhao, Wayne Xin and Hou, Yupeng and Pan, Xingyu and Yang, Chen and Zhang, Zeyu and Lin, Zihan and Zhang, Jingsen and Bian, Shuqing and Tang, Jiakai and Sun, Wenqi and others},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4722--4726},
  year={2022}
}
```



