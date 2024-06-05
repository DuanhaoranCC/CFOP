# CFOP

This repository is called SimGOP in the previous version, I renamed the model name in the camera-ready version.

This repository is for the source code of the journal Expert Systems With Applications paper "Contextual Features Online Prediction for Self-Supervised Graph Representation."

## Dependencies

```python
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

For node classification task:
```python
python main.py --dataset Cora
```

The `--dataset` argument should be one of [Cora, CiteSeer, PubMed, Com, Photo, Phy, CS, WikiCS].

## Cite
```
@article{duan2024contextual,
  title={Contextual features online prediction for self-supervised graph representation},
  author={Duan, Haoran and Xie, Cheng and Tang, Peng and Yu, Beibei},
  journal={Expert Systems with Applications},
  volume={238},
  pages={122075},
  year={2024},
  publisher={Elsevier}
}
```
