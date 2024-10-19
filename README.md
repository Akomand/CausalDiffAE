# Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models
This is the source code for the implementation of "Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models" (ECAI 2024)

Diffusion probabilistic models (DPMs) have become the state-of-the-art in high-quality image generation. However, DPMs have an arbitrary noisy latent space with no interpretable or controllable semantics. Although there has been significant research effort to improve image sample quality, there is little work on representation-controlled generation using diffusion models. Specifically, causal modeling and controllable counterfactual generation using DPMs is an underexplored area. In this work, we propose CausalDiffAE, a diffusion-based causal representation learning framework to enable counterfactual generation according to a specified causal model. Our key idea is to use an encoder to extract high-level semantically meaningful causal variables from high-dimensional data and model stochastic variation using reverse diffusion. We propose a causal encoding mechanism that maps high-dimensional data to causally related latent factors and parameterize the causal mechanisms among latent factors using neural networks. To enforce the disentanglement of causal variables, we formulate a variational objective and leverage auxiliary label information in a prior to regularize the latent space. We propose a DDIM-based counterfactual generation procedure subject to do-interventions. Finally, to address the limited label supervision scenario, we also study the application of CausalDiffAE when a part of the training data is unlabeled, which also enables granular control over the strength of interventions in generating counterfactuals during inference. We empirically show that CausalDiffAE learns a disentangled latent space and is capable of generating high-quality counterfactual images.


## Usage

### Training and evaluating 

1. Clone the repository

     ```
     git clone https://github.com/Akomand/CausalDiffAE.git
     cd CausalDiffAE
     ```
2. Create environment and install dependencies
   ```
   conda env create -f environment.yml
   ```
3. Create Dataset in ```image_datasets.py```
3. Specify Causal Adjacency Matrix A in ```unet.py```
   ```
   A = th.tensor([[0, 1], [0, 0]], dtype=th.float32)
   ```
4. Specify hyperparameters and run training script
   ```
    ./train_[dataset]_causaldae.sh
   ```
5. For classifier-free paradigm training, set ```masking=True``` in hyperparameter configs
6. To train anti-causal classifiers to evaluate effectiveness, run
   ```
   python [dataset]_classifier.py
   ```
7. For counterfactual generation, run the following script with the specified causal graph
   ```
    ./test_[dataset]_causaldae.sh
   ```
8. Modify ```image_causaldae_test.py``` to perform desired intervention and sample counterfactual

### Data acknowledgements
Experiments are run on the following datasets to evaluate our model:

#### Datasets
<details closed>
<summary>MorphoMNIST Dataset</summary>

[Link to dataset](https://github.com/dccastro/Morpho-MNIST)
</details>

<details closed>
<summary>Pendulum Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>CausalCircuit Dataset</summary>

[Link to dataset](https://developer.qualcomm.com/software/ai-datasets/causalcircuit)
</details>

## Citation

If you use our code or think our work is relevant to yours, we encourage you to cite this paper:

```bibtex
@inproceedings{
komanduri2024causaldiffae,
title={Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models},
author={Aneesh Komanduri and Chen Zhao and Feng Chen and Xintao Wu},
booktitle={Proceedings of the 27th European Conference on Artificial Intelligence},
year={2024}
}
```
