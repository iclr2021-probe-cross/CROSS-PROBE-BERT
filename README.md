# Cross-Probe-BERT
## Prepare dataset and models

### Dataset
MS coco dataset, you can download the testing features through this [link](https://www.dropbox.com/s/4b1yx5quegervw7/test_100feat.npy?dl=0) 

you can download the training features through this [link](https://www.dropbox.com/s/1wv3xs5rxdweagh/train_100feat.npy?dl=0)

then move your downloaded test_100feat.npy and train_100feat.npy to data/coco
### BERT Model
you can download the pretrained bert model provided by HuggingFace through this [link](https://www.dropbox.com/s/734z9hd6jgg2ier/pytorch_model.bin?dl=0)

then move your downloaded pytorch_model.bin file to ./bert fold
### Our Trained Model
COCO without pretraining [link](https://www.dropbox.com/s/73pyyxd8fanadtl/coco_nopretrain.pth.tar?dl=0) 

COCO with pretraining  [link](https://www.dropbox.com/s/r0tznnoq5i5u8tg/coco_pretrain.pth.tar?dl=0)
## Run Script
python train.py --batch_size 128 --num_epochs=31 --lr_update=15 --learning_rate=.00005
## Test Script
Test our trained model without pretraining:

CUDA_VISIBLE_DEVICES=0 python test.py --resume coco_nopretrain.pth

Test our trained model without pretraining:

CUDA_VISIBLE_DEVICES=0 python test.py --resume coco_pretrain.pth 


