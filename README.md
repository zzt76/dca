# Consistent Depth Estimation under Various Illuminations using Dilated Cross Attention

Official website for DCA and Vari dataset

## Abstract
***
In this work, we target at solving the problem of consistent depth estimation in complex scenes under various illumination conditions. The existing indoor datasets based on RGB-D sensors or virtual rendering have two critical limitations - sparse depth maps (NYU Depth V2) and non-photorealistic illumination (SUN CG, SceneNet RGB-D ). We propose to use internet-available 3D indoor scenes and manually tune their illuminations to render photorealistic RGB photos and their corresponding depth and BRDF maps, obtaining a new dataset called Various Illuminations (Vari). We propose a simple convolutional block named Dilated Cross Attention (DCA) by applying depthwise separable dilated convolution on encoded features to process global information and reduce parameters. Cross attention on these dilated features are performed to retain consistency of depth estimation under different illuminations. Our method is evaluated by comparing with state-of-the-art methods on Vari dataset and a significant improvement is observed quantitatively and qualitatively in our experimental results. We also conduct ablation study and finetune our model on NYU Depth V2 to validate the effectiveness of DCA block.

## Pretrained model
***
You can download the pretrained models on Vari and NYU dataset [here](https://1drv.ms/u/s!Al8Z5hpFSN2xgo4TG_mgioTtbLSRTg?e=t8XeqE).

## Inference
***
Download and move the pretrained weights to "./pretrained" directory.
### Predict RGB image from path or directory
```python
from infer import Inference

infer = Inference(pretrained_path='pretrained/dca_vari.pth', device=0)
# predict depth from an image tensor of size [b,c,h,w]
pred = infer.predict(im)

# predict depth from a RGB image
infer.predict_path(path='test_images/485_c1_SunMorning_Indoor_Environment0188.jpg')

# predict depth maps from RGB images of a directory
infer.predict_dir(im_dir='test_images', save_dir='results')
```

### Evaluation
Download Vari and NYU dataset and use the provided scripts in "scripts" folder to generate txt files.
```bash
# Evaluate on Vari dataset
python run.py --config experiments/test_vari.yml
# Evaluate on NYU Depth V2 dataset
python run.py --config experiments/test_nyu.yml
```

### Training
```bash
# Train on Vari dataset
python run.py --config experiments/train_vari.yml
# Train on NYU Depth V2 dataset
python run.py --config experiments/train_nyu.yml
```

## Vari Dataset
Coming soon...
