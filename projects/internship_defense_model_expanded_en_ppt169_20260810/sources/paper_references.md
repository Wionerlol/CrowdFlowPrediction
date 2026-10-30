# Verified primary paper references

1. Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*. IJCAI. https://arxiv.org/abs/1709.04875
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 30. https://papers.neurips.cc/paper/7181-attention-is-all-you-need
3. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR. https://arxiv.org/abs/1609.02907
4. Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*. IJCAI. https://arxiv.org/abs/1906.00121
5. Grigsby, J., Wang, Z., Nguyen, N., & Qi, Y. (2021). *Long-Range Transformers for Dynamic Spatiotemporal Forecasting*. arXiv:2109.12218. https://arxiv.org/abs/2109.12218
6. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. https://arxiv.org/abs/1312.6114
7. Xu, J., Wu, H., Wang, J., & Long, M. (2022). *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*. ICLR. https://arxiv.org/abs/2110.02642
8. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30. https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
9. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD. https://arxiv.org/abs/1907.10902

## Project-specific model naming note

`AGFormer` in this repository is the project's Adaptive Graph Transformer implementation. It combines temporal multi-head self-attention, a predefined graph branch, a learned adaptive adjacency branch, gated context fusion, and a parallel linear residual head. It is not presented as a direct reproduction of an external paper named AGFormer; its foundations are cited through Transformer, GCN, Graph WaveNet, STGCN, and the project's own implementation and ablation records.
