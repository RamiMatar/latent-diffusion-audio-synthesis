# Audio Latent Diffusion Models

This project builds a latent diffusion model for text to audio generation. Text-to-audio generation has recently become a popular topic with the advancements in training guided diffusion models. This project is largely based on the work proposed in the AudioLDM paper and their state of the art TTA system which can be trained in a self-supervised manner without needing audio-text data pairs.

# Primary References
1. Liu, Haohe, et al. "Audioldm: Text-to-audio generation with latent diffusion models." arXiv preprint arXiv:2301.12503 (2023).
2. Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
3. Wu, Yusong, et al. "Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.
4. Kong, Jungil, Jaehyeon Kim, and Jaekyoung Bae. "Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis." Advances in Neural Information Processing Systems 33 (2020): 17022-17033.
5. Di Giorgi, Bruno, Mark Levy, and Richard Sharp. "Mel Spectrogram Inversion with Stable Pitch." arXiv preprint arXiv:2208.12782 (2022).

## Usage - Evaluation and prompting

First, make sure to activate the conda env

```bash
conda env create -f environment.yml
conda activate tta
```
Make sure to download the model checkpoints and dataset (FMA_large)

```bash
wget https://os.unil.cloud.switch.ch/fma/fma_large.zip # 
```

To train the model:
```bash
python diffusion.py --batch_size 1 --dataset_path fma_large
```

To evaluate with text prompts
```bash
python eval.py --prompts "a flute playing a soft melody" "a piano song"
```

