# Reinforcement Learning for Solving Colored Traveling Salesman Problems: An Entropy-Insensitive Attention Approach

PyTorch implementation of "Reinforcement Learning for Solving Colored Traveling Salesman Problems: An Entropy-Insensitive Attention Approach" (GEIAM)


## Description


## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)


## Usage

First move to `PyTorch` dir. 

```
cd PyTorch
```

Then, generate the pickle file contaning hyperparameter values by running the following command.

```
python config.py
```

you would see the pickle file in `Pkl` dir. now you can start training the model.

```
python train.py -p Pkl/***.pkl
```

Plot prediction of the pretrained model
(in this example, batch size is 128, number of customer nodes is 50)

```
python plot.py -p Weights/***.pt(or ***.h5) -b 128 -n 50
```

You can change `plot.py` into `plot_2opt.py`.  
  
2opt is a local search method, which improves a crossed route by swapping arcs.  

  

## Reference
* https://github.com/wouterkool/attention-learn-to-route
