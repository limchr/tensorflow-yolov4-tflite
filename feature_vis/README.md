# Using Deep Dream:
```shell
python approach_lucid \
[--neurons / -n, default "output_neurons[5][0][6][6][0][5]", add multiple strings with spaces] \
[--img / -i, default "None", choice=["Random", "None"] or a path seperated by spaces to an img (e.g) "data" "img.jpg"] \
[--steps / -s, default 1500, Number of steps to train on] \
[--step_size / -sz, default 0.05, 1e-08 < sz < 0.99~, step size during optimization] \
[--save_every / -se] TODO:Currently does nothing, change to saving intermediate images in code!
[--total_variance / -tv, default -2.5e-10, 0 < tv < 0.1]
```






## Files
### detect_steps.py
Seperated functions for the detection in an image with a trained model. Preprocess_image works with a batch, the other functions not.

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
