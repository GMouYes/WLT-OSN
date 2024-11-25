# WLT-OSN
Link for ICWSM 2024 paper: [Wildlife Product Trading in Online Social Networks: A Case Study on Ivory-Related Product Sales Promotion Posts](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/ICWSM/article/view/31375&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=10571559063033688816&ei=WQFFZ63eCu-Vy9YPubC3QQ&scisig=AFWwaebZNg9SKVtOi02nv5f1YyDk)

Sharing code and dataset to encourage future work. 

## Citing our work
```
@inproceedings{mou2024wildlife,
  title={Wildlife Product Trading in Online Social Networks: A Case Study on Ivory-Related Product Sales Promotion Posts},
  author={Mou, Guanyi and Yue, Yun and Lee, Kyumin and Zhang, Ziming},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={18},
  pages={1096--1109},
  year={2024}
}
```

## Folders
### Data
We are excited to share our dataset with the public to encourage more research focusing on preventing illicit wildlife trafficking behaviors.

Data contains our shared new dataset named as ``share_data.csv``, containing the necessary information around the original posting ids and its cleaned/masked text information. One may retrieve the original text from Twitter given the information, or simply leverage our provided cleaned version.

Together with the csv file, we also provide the retrieved images under ``data/images/``, each image is formatted based on the ids in the csv file. For each post, there may exist 0 to at most 4 images.
Researchers may leverage their preferred off-the-shelf methods to extract OCR information from images. 

It is very important to note, that we are sharing the full labelled dataset, which is much larger than the sampled (sub)dataset for benchmarking. One may sample from the large dataset considering the highly imbalanced nature.
### Code
Thank you for the interest in our experiment code. 

Code contains sample .py code and bash shells for BERT+ViT model, running on text+image+ocr results.

One may adapt the bash shell and scripts to easily run other variations. They should be mostly similar and require minimal adaptation effort for ones familiar with pytorch and coding with language/vision models.
### config, logs, and output
We provide a sample config file to run coorperate with the provided code runnign on BERT+ViT.

To run our code, please follow these steps:
- Ensure sufficient packages installed.
- Ensure sufficient GPU vram and ram on the machine.
- Based on your research need, sample and split the shared data into train/valid/test csv files.
- Based on your research need, adjust the configuration files. For example: whether or not employ descriptions/ocr as extra information, is up to the researcher and can be controlled in the config
- Feel free the adjust any of our provided sample code. Once you have prepared your desired data, run crossValidation.sh for training, run valid.sh for inferring on valid set so you may recalibrate classification threshold, run inference.sh to inference on test set so you may come to final result.

A more formatted data repository with extensive details are to be reveiled upon paper acceptance.
All data and code provided in the repository are solely for research purpose. Do not share for commercial usage.

