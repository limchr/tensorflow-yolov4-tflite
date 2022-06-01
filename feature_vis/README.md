# Using Deep Dream:
```shell
python feature_vis/approach_lucid.py \
[-neurons / -n, default "output_neurons[5][0][6][6][0][5]", String of Neurons to Optimize, seperated by spaces, default is output_neurons[5][0][6][6][0][5]] \
[-img / -i, default "None", String of path to single input image to use via os.path, seperated by spaces, default is a gray image] \
[-steps / -s, default 1500, Expects integer, Number of steps for neuron optimization] \
[-step_size / -sz, default 0.05, Expects float as step size] \
[-save_every / -se, default 100, Expects int, steps after which an intermediate image is saved] \
[-file_name / -fn, default "deep_dream_test", Expects string, file name for saving images] \
[-file_path / -fp, default "", Expects list of strings seperated by spaces, file directory for saving images] \
# Regularizations
[-total_variance / -tv, default -0.000000025, Expects float, Total variance regularization] \
[-lasso_1 / -l1, default 0.2, Expects float, l1 regularization] \
[-lasso_2 / -l2, default 2.0 Expects float, l2 regularization] \
[-padding / -p, default 1, Expects int, pad regularization] \
[-color_difference / -c, default 0, Expects float, color hist regularization, pairs well with input img] \
[-repredouce, defaulte False, Flag that enables reproducability mode for random numbers (tf set seed with 2022)]

```

## Note on neurons

Syntax "output_neurons[A][B][C][D][E][F]" where:

- A -> YOLO head (refer to https://limchr.github.io/yolo_visualization/)
  - 0: 52 x 52 x 3 bounding boxes
  - 1: 52 x 52 x 1 bounding boxes
  - 2: 52 x 52 x 3 bounding boxes
  - 3: 26 x 26 x 1 bounding boxes
  - 4: 13 x 13 x 3 bounding boxes
  - 5: 13 x 13 x 1 bounding boxes
- B -> Anchor Box choice if A in [0,2,4], else 0
- C -> X Index of Anchor
- D -> Y Index of Anchor
- E -> Choice of bounding box style specific to head (0-3)
- F -> Neuron to maximize XYHWC + classes (0-85)

Due to the nature of output_neurons, one can easily combine multiple outputs for different anchor boxes and even combine
different heads easily.

## Files

### detect_steps.py

Seperated functions for the detection in an image with a trained model. Preprocess_image works with a batch, the other
functions not.

### opt_helper.py
Various functions to insert a layer into an existing model while leaving configurations and weights of existing layers untouched.  
Most important is create_model_plus_one, which does the job.

### config.py
Configurations for individual machine. Could be changed to Flags later.

### optimize_output.py
Contains a function to transform the output of decode_train into the format of decode.
Main function to  
a) build new model with additional layer,  
b) get the model and compile it with custom loss,  
c) train the model and save it

### random_training_data
Contains multiple (200 at the moment) images from the original dataset.
