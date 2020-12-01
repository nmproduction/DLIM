# DLIM Image domain transfer project

This project is heavily based on pix2pix, we in fact started with the repository of pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and then modified and added some parts to try our own experiments. Therefore, before getting started it is important to set up all requirements for pix2pix thus part of the readme of the original repository is listed below, after which additional instructions will be added.

Citations for pix2pix:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)

## Talks and Course
pix2pix slides: [keynote](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.key) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.pdf),
CycleGAN slides: [pptx](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pptx) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pdf)

CycleGAN course assignment [code](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip) and [handout](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) designed by Prof. [Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) for [CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/) "Intro to Neural Networks and Machine Learning" at University of Toronto. Please contact the instructor if you would like to adopt it in your course.

## Colab Notebook
TensorFlow Core CycleGAN Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)

TensorFlow Core pix2pix Tutorial: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)

PyTorch Colab notebook: [CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) and [pix2pix](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)

ZeroCostDL4Mic Colab notebook: [CycleGAN](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/CycleGAN_ZeroCostDL4Mic.ipynb) and [pix2pix](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/pix2pix_ZeroCostDL4Mic.ipynb)

## Other implementations
### CycleGAN
<p><a href="https://github.com/leehomyc/cyclegan-1"> [Tensorflow]</a> (by Harry Yang),
<a href="https://github.com/architrathore/CycleGAN/">[Tensorflow]</a> (by Archit Rathore),
<a href="https://github.com/vanhuyz/CycleGAN-TensorFlow">[Tensorflow]</a> (by Van Huy),
<a href="https://github.com/XHUJOY/CycleGAN-tensorflow">[Tensorflow]</a> (by Xiaowei Hu),
<a href="https://github.com/LynnHo/CycleGAN-Tensorflow-Simple"> [Tensorflow-simple]</a> (by Zhenliang He),
<a href="https://github.com/luoxier/CycleGAN_Tensorlayer"> [TensorLayer1.0]</a> (by luoxier),
<a href="https://github.com/tensorlayer/cyclegan"> [TensorLayer2.0]</a> (by zsdonghao),
<a href="https://github.com/Aixile/chainer-cyclegan">[Chainer]</a> (by Yanghua Jin),
<a href="https://github.com/yunjey/mnist-svhn-transfer">[Minimal PyTorch]</a> (by yunjey),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/CycleGAN">[Mxnet]</a> (by Ldpe2G),
<a href="https://github.com/tjwei/GANotebooks">[lasagne/Keras]</a> (by tjwei),
<a href="https://github.com/simontomaskarlsson/CycleGAN-Keras">[Keras]</a> (by Simon Karlsson),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Oneflow-Python/CycleGAN">[OneFlow]</a> (by Ldpe2G)
</p>
</ul>

### pix2pix
<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (by mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (by taey16)
</p>
</ul>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix train/test
- Download a pix2pix dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`.

- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.

### Apply a pre-trained model (CycleGAN)
- You can download a pretrained model (e.g. horse2zebra) with the following script:
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- To test the model, you also need to download the  horse2zebra dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For pix2pix and your own models, you need to explicitly specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model. See this [FAQ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#runtimeerror-errors-in-loading-state_dict-812-671461-296) for more details.

### Apply a pre-trained model (pix2pix)
Download a pre-trained model with `./scripts/download_pix2pix_model.sh`.

- Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3) for all the available pix2pix models. For example, if you would like to download label2photo model on the Facades dataset,
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```
- Download the pix2pix facades datasets:
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Then generate the results using
```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
- Note that we specified `--direction BtoA` as Facades dataset's A to B direction is photos to labels.

- If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use `--model test` option. See `./scripts/test_single.sh` for how to apply a model to Facade label maps (stored in the directory `facades/testB`).

- See a list of currently available models at `./scripts/download_pix2pix_model.sh`

## [Docker](docs/docker.md)
We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](docs/docker.md).

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Pull Request
You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/).
Please run `flake8 --ignore E501 .` and `python ./scripts/test_before_push.py` before you commit the code. Please also update the code structure [overview](docs/overview.md) accordingly if you add or remove files.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Other Languages
[Spanish](docs/README_es.md)

## Related Projects
**[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)**<br>
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)|
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**<br>
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper [Collection](https://github.com/junyanz/CatPapers).

## Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).


# DLIM Image Domain Transfer additional requirements

Please install torch-fidelity, since it is used in many places to evaluate our results. 
https://github.com/toshas/torch-fidelity
the easiest way to install it is using pip.

In the following experiments the fid score and inception scores are used extensively. To get these metrics one simply has to run two commands in python. Below you can see an example. To adapt it to different networks only the path of the test results must be adjusted to the previously trained network.

```
from torch_fidelity import calculate_metrics

#runs the inception pretrained classifier and looks how easily it can distinguish different stuff in the generated images
#input1 = first path to sample of images
#input2 = second path to sample of images
#cuda = gpu usage
#isc = inception score
#kid = kernel inception distance
#fid = frechet inception distance
pretrained_metrics = calculate_metrics("./results/facades_label2photo_pretrained/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
trained_metrics = calculate_metrics("./results/perceptual/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)

print("pretrained metrics", pretrained_metrics)
print("trained metrics for perceptual", trained_metrics)
```



# Batch size experiments
Run the following commands in the terminal to train pix2pix with different images per batch and test them. The results should look as below:
```
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch1 --direction BtoA --batch_size 1
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch4 --direction BtoA --batch_size 4
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch16 --direction BtoA --batch_size 16
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch64 --direction BtoA --batch_size 64
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch256 --direction BtoA --batch_size 256

python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch1
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch4
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch16
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch64
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch256
```
Below you can see the results that we achieved together with the score on inception and fid.
<img src='imgs/batch_size_experiments.png' width=600><img src='imgs/batch_size_experiments_graph.png' width=600>

# Perceptual loss
We used parts of the SPADE network (https://github.com/NVlabs/SPADE) to swap an L1 loss for a higher level perceptual loss. For that we added another class to the model folder. You can try it out by simply running it in the terminal`
```
python ./train.py --dataroot ./datasets/facades --model perceptual --name perceptual --direction BtoA 
python ./test.py --dataroot ./datasets/facades --direction BtoA --model perceptual --name perceptual 
```
Below you can see the results that we achieved together with the score on inception and fid.
<img src='imgs/perceptual_loss_facades.png' width=600> <img src='imgs/perceptual_loss_facades_score.png' width=600>
perceptual_loss_facades_score

# Adversarial losses
We once again oriented ourselves on SPADES and implemented different losses that also compare the fake image directly to the correct one. Instead of L1 once can try LSGAN, WGAN and Hinge WGAN. The adjusted model (with an option loss) is saved in models/adversarial_losses_model.py
```
python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_og --direction BtoA --gan_mode original
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_og

python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_ls --direction BtoA --gan_mode ls
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_ls

python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_hinge --direction BtoA --gan_mode hinge
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_hinge 

python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_w --direction BtoA --gan_mode w
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_w 
```
To then test the fidelity score we run
```
from torch_fidelity import calculate_metrics

#runs the inception pretrained classifier and looks how easily it can distinguish different stuff in the generated images
#input1 = first path to sample of images
#input2 = second path to sample of images
#cuda = gpu usage
#isc = inception score
#kid = kernel inception distance
#fid = frechet inception distance
cross_entropy_metric = calculate_metrics("./results/adv_og/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
least_squares_metric = calculate_metrics("./results/adv_ls/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
hinge_metric = calculate_metrics("./results/adv_hinge/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
wasserstein_metric = calculate_metrics("./results/adv_w/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)

print("cross_entropy_metric", cross_entropy_metric)
print("least_squares_metric", least_squares_metric)
print("hinge_metric", hinge_metric)
print("wasserstein_metric", wasserstein_metric)
```

Our results are as follows:
<img src='imgs/adversarial_losses_scores.png' width=600><img src='imgs/adversarial_losses_examples.png' width=600>


# CelebAMask-HQ

[[Paper]](https://arxiv.org/abs/1907.11922) [[Demo]](https://www.youtube.com/watch?v=T1o38DFalWs)  

![image](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/images/front.jpeg)

**CelebAMask-HQ** is a large-scale face image dataset that has **30,000** high-resolution face images selected from the CelebA dataset by following CelebA-HQ. Each image has segmentation mask of facial attributes corresponding to CelebA.

The masks of CelebAMask-HQ were manually-annotated with the size of **512 x 512** and **19 classes** including all facial components and accessories such as skin, nose, eyes, eyebrows, ears, mouth, lip, hair, hat, eyeglass, earring, necklace, neck, and cloth. 

CelebAMask-HQ can be used to **train and evaluate algorithms of face parsing, face recognition, and GANs for face generation and editing**.

* If you need the identity labels and the attribute labels of the images, please send request to the [CelebA team](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

* Demo of interactive facial image manipulation

![image](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/images/demo.gif)

## Sample Images

![image](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/images/sample.png)

## Face Manipulation Model with CelebAMask-HQ
CelebAMask-HQ can be used on several research fields including: facial image manipulation, face parsing, face recognition, and face hallucination. We showcase an application on interactive facial image manipulation as bellow:

* Samples of interactive facial image manipulation

![image](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/images/sample_interactive.png)

## CelebAMask-HQ Dataset Downloads
* Google Drive: [downloading link](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv)
* Baidu Drive: [downloading link](https://pan.baidu.com/s/1wN1E-B1bJ7mE1mrn9loj5g)

## Related Works
* **CelebA** dataset:<br/>
Ziwei Liu, Ping Luo, Xiaogang Wang and Xiaoou Tang, "Deep Learning Face Attributes in the Wild", in IEEE International Conference on Computer Vision (ICCV), 2015 
* **CelebA-HQ** was collected from CelebA and further post-processed by the following paper :<br/>
Karras et. al, "Progressive Growing of GANs for Improved Quality, Stability, and Variation", in Internation Conference on Reoresentation Learning (ICLR), 2018

## Dataset Agreement
* The CelebAMask-HQ dataset is available for **non-commercial research purposes** only.
* You agree **not to** reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
* You agree **not to** further copy, publish or distribute any portion of the CelebAMask-HQ dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

## Related Projects using CelebAMask-HQ
* [SPADE-TensorFlow](https://github.com/taki0112/SPADE-Tensorflow)
* [FaceParsing-PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.
```
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
**

## Recreating the results
 

 **Training the pix2pix model on the cityscapes dataset**

 1. Using pip install the requirements in requirements.txt with pip install -r requirements.txt
 2.  follow the instructions in conda-env.txt to install the rest of the required packages using anaconda (you can also use the import button in anaconda navigator and select conda-env.txt)
 3. for training: in the command line run `python train.py --n_epochs 100 --n_epochs_decay 100 --dataroot ./datasets/cityscapes --name cityscapes_pix2pix --model pix2pix --direction BtoA`

the dataset is contained already in the repo

 4. for testing run `python test.py --dataroot ./datasets/cityscapes --direction BtoA --model pix2pix --name cityscapes_pix2pix`


**Using the GUI for testing the model:**
 1. Execute the file MaskGAN_demo\demo_cityscapes.py using the command `python MaskGAN_demo\demo_cityscapes.py`
 2. in the GUI select a input image form the images folder in the same directory and the corresponding labels
