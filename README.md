# DIP-proj-DepthEstimation

## Directories

* ./model
  * block.py
    * define the encoder(now only resnet50), mff, decoder and refinement block, which are modules of the model
  * model.py
    * define the model, utilizing the blocks in block.py
  * resnet_module.py
    * self-defined resnet for encoding blocks
      * credit to https://github.com/JunjH/

* nyu2_train
  * only part of the dataset, used to check whether the code is runnable

* load_data.py
  * load the data by making pairs of **path**, the output is a list of pair \[x\_tr\_path, y\_tr\_path\]

* loss.py
  * define the gradient computation and loss computation

* utils.py
  * currently, only loading and saving the params of model is defined

* train.py
  * define the training procedure

