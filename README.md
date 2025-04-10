# A Differential Geometric View and Explainability of GNN on Evolving Graphs
This is the Pytorch Implementation of [A Differential Geometric View and Explainability of GNN on Evolving Graphs](https://openreview.net/forum?id=lRdhvzMpVYV)
# Train a Model
You need to train a model first. To train the model, use th following command, replacing $(dataset) with the name of dataset. **Chi**, **NYC**, and **Zip** are the node classification datasets. **Bitcoinalpha**, **bitcoinotc**, and **UCI** are the link prediction datasets. **Mutag** are the graph classification datasets. Replace $(task) with the node, link or graph.
```bash
python train_GCN_$(task).py --data $(dataset)
```
# Provide an explanation
Once you have a  model, the next step is to provide an explanation. You can find the important paths as the explanations. Use the following command to obtain the explanation,replaceing $(task) with the node, link or graph. Run the following command can obtain the data in Figure 2 and Figure 3.
```bash
python path_explain_$(task).py --data $(dataset)
```
# Citation
```bash
@article{liu2023convex,
  title={A Differential Geometric View and Explainability of GNN on Evolving Graphs},
  author={Yazheng Liu, Xi Zhang and Sihong Xie},
  conference={International Conference on Learning Representations},
  year={2023}
}
```
