<h1 align='center'>Hallo4: High-Fidelity Dynamic Portrait Animation via Direct Preference
Optimization</h1>

<div align='center'>
    <a href='https://cuijh26.github.io/' target='_blank'>Jiahao Cui</a><sup>1*</sup>&emsp;
    <a href='https://github.com/cbyzju' target='_blank'>Baoyou Chen</a><sup>1*</sup>&emsp;
    <a href='https://github.com/xumingw' target='_blank'>Mingwang Xu</a><sup>1*</sup>&emsp;
    <a href='https://github.com/NinoNeumann' target='_blank'>Hanlin Shang</a><sup>1</sup>&emsp;
    <a href='https://github.com/Shr1ke777' target='_blank'>Yuxuan Chen</a><sup>1</sup>&emsp;
</div>
<div align='center'>
    <a href='https://orcid.org/0000-0001-6977-9989' target='_blank'>Yun Zhan</a><sup>1</sup>&emsp;
    <a href='https://orcid.org/0000-0002-6833-9102' target='_blank'>Zilong Dong</a><sup>5</sup>&emsp;
    <a href='https://orcid.org/0000-0001-9866-4291' target='_blank'>Yao Yao</a><sup>4</sup>&emsp;
    <a href='https://jingdongwang2017.github.io/' target='_blank'>Jingdong Wang</a><sup>2</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>1,3✉️</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Fudan University&emsp; <sup>2</sup>Baidu Inc&emsp; <sup>3</sup>Shanghai Innovative Institute&emsp;
</div>

<div align='center'>
    <sup>4</sup>Nanjing University&emsp; <sup>5</sup>Alibaba Group&emsp;
</div>

<br>
<div align='center'>
    <a href='https://github.com/fudan-generative-vision/hallo4'><img src='https://img.shields.io/github/stars/fudan-generative-vision/hallo4.svg'></a>
    <a href='https://xyz123xyz456.github.io/hallo4/#/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/pdf/2505.23525'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/fudan-generative-ai/hallo4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://huggingface.co/datasets/cuijh26/hallo4_data'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow'></a>
</div>
<div align='Center'>
    <i><strong><a href='https://asia.siggraph.org/2025/' target='_blank'>SIGGRAPH Asia 2025</a></strong></i>
</div>
<br>

## 📸 Showcase

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/37f2fa64-3c46-4384-8a2c-b40d5a0ed21c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/35a06195-60cd-4917-a70e-930aa5a14241" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/3457a7c5-cbd3-4e1c-a5c5-04604965761c" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d22715b4-e411-4346-83a9-f21510746e42" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/fccffb71-8710-4f24-94ab-d249582b1b56" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/86532666-01bc-415b-95d7-f89a5606f14b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


## ⚙️ Installation

- System requirement: Ubuntu 20.04/Ubuntu 22.04, Cuda 12.1
- Tested GPUs: H100

Download the codes:

```bash
  git clone https://github.com/fudan-generative-vision/hallo4
  cd hallo4
```

Create conda environment:

```bash
  conda create -n hallo python=3.10
  conda activate hallo
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
```

Besides, ffmpeg is also needed:

```bash
  apt-get install ffmpeg
```

### 📥 Download Pretrained Models

You can easily get all pretrained models required by inference from our [HuggingFace repo](https://huggingface.co/fudan-generative-ai/hallo4).

Using `huggingface-cli` to download the models:

```shell
cd $ProjectRootDir
pip install "huggingface_hub[cli]"
huggingface-cli download fudan-generative-ai/hallo4 --local-dir ./pretrained_models
```

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- hallo4
|   `-- model_weight.pth
|-- Wan2.1_Encoders
    |-- Wan2.1_VAE.pth
    |-- models_t5_umt5-xxl-enc-bf16.pth
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- wav2vec/
    `-- wav2vec2-base-960h/
        |-- config.json
        |-- feature_extractor_config.json
        |-- model.safetensors
        |-- preprocessor_config.json
        |-- special_tokens_map.json
        |-- tokenizer_config.json
        `-- vocab.json
```

### 🛠️ Prepare Inference Data

Hallo4 have some specicial requirements on inference data due to limitation of our training:
1. Reference image should have aspect ratio between 1:1 and 480:832.
2. Driving audio must be in WAV format.
3. Audio must be in English since our training datasets are only in this language.
4. Ensure the vocals of audio are clear; background music is acceptable.

### 🎮 Run Inference
To run a simple demo, just use our provided shell ```bash inf.sh```

<!-- Prepare the inference list with tool script
```bash
python -m prepare_case_list.py
```
Then start inference with follow command
```bash
python -m vace.vace_wan_inference --inference_list xxx
``` -->


## ⚠️ Social Risks and Mitigations

The development of portrait image animation technologies driven by audio inputs poses social risks, such as the ethical implications of creating realistic portraits that could be misused for deepfakes. To mitigate these risks, it is crucial to establish ethical guidelines and responsible use practices. Privacy and consent concerns also arise from using individuals' images and voices. Addressing these involves transparent data usage policies, informed consent, and safeguarding privacy rights. By addressing these risks and implementing mitigations, the research aims to ensure the responsible and ethical development of this technology.

## 🤗 Acknowledgements

This model is a fine-tuned derivative version based on the **WAN2.1-1.3B** model. WAN is an open-source video generation model developed by the WAN team. Its original code and model parameters are governed by the [WAN LICENSE](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt).

As a derivative work of WAN, the use, distribution, and modification of this model must comply with the license terms of WAN.

