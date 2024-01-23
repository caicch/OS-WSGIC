# Top-Down Framework for Weakly Supervised Grounded Image

## Requirements
- Python 3.8
- Pytorch 1.11

## Prepare data
1. Download [coco-caption] and [cider] evaluation from https://github.com/ruotianluo/coco-caption/tree/dda03fc714f1fcc8e2696a8db0d469d99b881411 and https://github.com/ruotianluo/cider/tree/dbb3960165d86202ed3c417b412a000fc8e717f3 
2. download and place the [Flickr30k reference file](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ERf4vteh7AdMmpR5jCc2ve4BNmZJ8EfY8LJVe4D3KCR4oQ?e=8qNj1W) under coco-caption/annotations. Also, download [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) for grounding evaluation and place the uncompressed folder under the tools/ directory.
3. Download the *Flickr30k-Entities* raw RGB image form http://hockenmaier.cs.illinois.edu/DenotationGraph/ and place it to data/.
4. Download the relation classes label from https://drive.google.com/file/d/1ZDST8PXhoFb_x_oZxlTSA7TPJus_F1xe/view?usp=share_link
5. Download the preprocessd dataset from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/Ea0HzFuNDGNPmmTxBTVjfbwBp9ZGhIAyyQylATXV735eUA?e=yEEaI6) and extract it to data/.
6. For *Flickr30k-Entities*, please download bottom-up visual feature extracted by Anderson's [extractor](https://github.com/peteanderson80/bottom-up-attention) (Zhou's [extractor](https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch)) from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/EWKJu8TLXtVPu5h3EnNRWo4BfWs_3WIBfoXXJPWFoIS5kA?e=IFSR8Q) ( [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ES446ZSwHCZAqiPjXxXW2twB_jMa_GmAiyuOUnEsNSWeUw?e=6u3pnF)) and place the uncompressed folders  under data/flickrbu/. 


## Training
1.  In the first training stage, run like
```
python train_WSGIC.py --id WSGIC --caption_model GIC --input_json data/flickrtalk.json --image_path data/flickr30k-images/ --input_fc_dir data/flickrbu/flickrbu_fc --input_att_dir data/flickrbu/flickrbu_att  --input_box_dir data/flickrbu/flickrbu_box  --input_label_h5 data/flickrtalk_label.h5 --input_label_h5_predicates data/flickrtalk_relation_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log/WSGIC --save_checkpoint_every 3000 --val_images_use -1 --max_epochs 20 
```

2. In the second training stage, run like

```
python train_WSGIC.py --id sc-WSGIC --caption_model GIC --image_path data/flickr30k-images/ --input_json data/flickrtalk.json --input_fc_dir data/flickrbu/flickrbu_fc --input_att_dir data/flickrbu/flickrbu_att  --input_box_dir data/flickrbu/flickrbu_box  --input_label_h5 data/flickrtalk_label.h5 --input_label_h5_predicates data/flickrtalk_relation_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log/WSGIC --checkpoint_path log/sc-WSGIC --save_checkpoint_every 3000 --language_eval 1 --val_images_use -1 --self_critical_after 20  --max_epochs 40
```

## evaluation
```
python eval.py  --model  ./log/WSGIC/model.pth    --infos_path   ./log/WSGIC/infos_WSGIC.pkl  --dataset flickr   --split  train  --eval_att  1   --att_supervise  1  --batch_size 1  --beam_size  1 --thresholding 0.05
python eval.py  --model  ./log/sc_WSGIC/model.pth    --infos_path   ./log/sc_WSGIC/infos_WSGIC.pkl  --dataset flickr   --split  train  --eval_att  1   --att_supervise  1  --batch_size 1  --beam_size  1 --thresholding 0.06

```
the code will be released upon publications
```
