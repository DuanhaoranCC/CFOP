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
@article{DBLP:journals/eswa/DuanXTY24,
  author       = {Haoran Duan and
                  Cheng Xie and
                  Peng Tang and
                  Beibei Yu},
  title        = {Contextual features online prediction for self-supervised graph representation},
  journal      = {Expert Syst. Appl.},
  volume       = {238},
  number       = {Part {C}},
  pages        = {122075},
  year         = {2024},
}
```
