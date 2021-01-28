# Explore
only focus on pytorch code of vision part.

## run code
check readme.md
## code structure
```
-data
-experiments
  -base_model
  -learning_rate
    -learning_rate_0.001
    -learning_rate_0.01
-model
    -model.py
    -data_loader.py
-train.py
-evaluate.py
-build_dataset.py
-search_hyperparameters.py
-synthesize_results.py
-utils.py
```

## code profiling
- build the dataset; split raw data into train/val/test and size images.
- load the dataset to gain (X,Y) tensors. In particular, we need obtain dataset and itertor(dataloader) for pytorch.
- model, loss and metrics; define how to map X to Y'(predictions), loss function, and metrics.
- training and evaluating;
- hyper-parameters tuning; train with multiple settings over 1 or more hyper-parameters.
- synthesis; summarize best metrics for all experiments.
- visualization;

### build the dataset
- goal; preprocess raw data(images)(e.g. save into desired image sizes) and split dataset into train/val/test.
- raw data structure and files;
```
SIGNS
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

- `build_dataset.py`; check schematic code analysis.
```
argparse arguments: data_dir, output_dir
obtain image filenames for training, testing 
split training into training and val set using training filenames
for split in [train/val/test]
    resize and save each file in split to output_dir
```

- obtained dataset stucture; **resize images and train/val/test split**
```
SIGNS_64x64
    train_signs/
        0_IMG_5864.jpg
        ...
    val_signs/
        0_IMG_5942.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

### load the dataset

- goal; load the dataset files to gain (X,Y) tensors for training. In particular, we need obtain dataset and itertor(dataloader) for pytorch.

- `data_loader.py`; check schematic code analysis.
```
# SIGNSDataset
create the dataset object inheriting torch.utils.data.Dataset
- override __init__ method; need define attributes of the dataset which can help link physical files to tensors. typical attributes are filenames for a particular train/val/test set, dataset name, labels, transform(need declare torchvision tranform ahead of time.)
- override __len__ method;
- override __getitem__ method; get (image,label) where image is loaded from a local physcial file possibly w. transforms and label is the category id.

# data_loader
create loader for each type of set(i.e. train/val/test)
dataloaders ={} # elegant data structure
for split in [train/val/test]
    if split in types
        get set path
        create the dataloader for train and val/test set(note: train need tranform operation(i.e. data augmentation) while val/test no need)
```

- ouput: dataloader for train and val/test set; each dataloader is a set of collection(iterator) which allows to iterate examples with ease.

```
dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], args.data_dir, params)
train_dl = dataloaders['train']
val_dl = dataloaders['val']
test_dl = dataloaders['test']
```


### model
- goal;define how to map X to Y'(predictions) i.e. build the computation graph, loss function, and metrics. Mathmatically, we need define two functions Y' = f (W|X,Y), Loss = Loss(Y',Y). Typically, it consists of many layers or blocks.
- `net.py`; check schematic code analysis.
```
# Net
create the model object inheriting torch.nn.Module
- override __init__ method; need define layers, and other settings.
- override forward method; define how to operate over an batch input i.e. how the input tensor is flowed to gain the prediction scores. Specially, input--layer1-layer2-...--scores

# loss_fn
compute the distance between Y'(predicted labels) and Y(i.e. labels); we can use built-in pytorch losses(i.e. torch.nn.loss.MSELoss)

# metrics
define which metrics to use, e.g. accuracy, precision, recall, f1score
```

- ouput: a model which maps batched input to predicted scores.

```
model = net.Net(params).cuda() if params.cuda else net.Net(params)
```

### train and evaluate

- goal; find optimal parameters for the training set where the model's loss (the distance between Y' and Y is smallest). Mathmatically, W = argmin(Loss(W|X,Y)). Algorithmically, the model parameters will be gained using gradient descent(GD) to gradually update the values of parameters. For GD, gradients of parameters are needed which will be computed using backprop(already implemnted in pytorch autograd)

- `train.py` and `evaluate.py`; check schematic code analysis.

train.py
```
argparse arguments: data_dir, model_dir(experimental setting dir), restore_dir(checkpoint dir)

load model config(e.g. hyperparameters) from model_dir's params.json(Params in utils.py)
set up logging object to serialize status of training to a file (check set_logger() in utils.py, which creat logger object and add filehandle to file and console)

create dataloaders for train/val/test

define the model, loss, metrics and optimizer

training and evaluating loop
if restore_file
    restore checkpoint
for epoch in range(epochs)
    (train_one_epoch)
    for X_batch, Y_batch in train_dataloader
        move to gpu
        Y'=model(X)
        loss=loss_fn(Y',Y)
        loss.backward() # compute grads
        gradient descent(optimizer.step(),need optim.zero_grad())
        store batch metrics to a dict and append to summary
        compute batch loss
        update loss for 1 batch
    compute mean metrics(acc,etc) for 1 epoch

    (evaluate_one_epoch)
    for X_batch, Y_batch in val_dataloader
        move to gpu
        Y'=model(X)
        loss=loss_fn(Y',Y)
        NOTE: no gradient update
        store batch metrics to a dict and append to summary
        compute batch loss
        update loss for 1 batch
    compute mean metrics(acc,etc) for 1 epoch
    determine whether obtain best accuaracy(or other metrics)
    if yes, save otherwise ignore
```

evaluate.py (similar logic to train.py but no learning proce(i.e. updating parameters))

- ouput: a model with optimal weights, logs, and pretrained weights serialized to local files.

### hyperparameters tuning
- goal; set different hyperparameter combination to find which one gain the best metric performance.

- `search_hyperparams.py`
```
argparse arguments: parent_dir(e.g. experiment/learning_rate) and data_dir(e.g. data/xx_SIGNS)
load parent_dir params.json
define search values for hyperparameters
for value in hyperparameter_values
    make different model hyperparameter settings (e.g. different learning rates)
    lauch_training_job(which can execute train.py w. the model_dir(constructed based) and data_dir)
```

- ouput: several training with different settings on 1 hyperparameter(e.g. learning rate), multiple folders (e.g. experiments/learning_rate/ has several folders like learning_rate_value1, learning_rate_value2,...)

### summarize metrics of all experiments 

- goal; extract best metrics(metrics_eval_best_weight.json) for all experiments and output a table.

- `synthesize_results.py`
```
argparse arguments: parent_dir
aggregate metrics
- search parent_dir metrics_eval_best_weight.json
- search its sub-dirs(here use the recursive function)
- return a metrics dict
create a table using tabulate package which require table headers and table(i.e. subdir+[*values]) 
write to a file(results.md) or write to excel file using pandas.
```

- ouput:

# Exploit
## Architecture Style Cls project

- [x] basic
- [x] DA and Transfer learning
- [x] analysis

## PointNet adapt

## RandLA-Net project(for PC seg paper)


## KPConv project(for PC cls paper)
