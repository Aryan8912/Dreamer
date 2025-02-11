# Dreamer

## Code Structure
Code structure is similar to original work by Danijar Hafner in Tensorflow

`dreamer.py`  - main function for training and evaluating dreamer agent

`utils.py`    - Logger, miscallaneous utility functions

`models.py`   - All the networks for world model and actor are implemented here

`replay_buffer.py` - Experience buffer for training world model

`env_wrapper.py`  - Gym wrapper for Dm_control suite

All the hyperparameters are listed in main.py and are avaialble as command line args.

#### For training
`python dreamer.py --env 'walker-walk' --algo 'Dreamerv1' --exp 'default_hp' --train`
#### For Evaluation
`python dreamer.py --env 'walker-walk' --algo 'Dreamerv1' --exp 'eval' --evaluate --restore --checkpoint_path '<your_ckpt_path>'`
