# Dreamer
<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/c428c97a-3beb-4279-b626-ef3074940b14" /> <img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/f6b6c151-6fe3-41f4-952f-e4769fb026a5" />
<img width="1952" height="495" alt="image" src="https://github.com/user-attachments/assets/8ab5d427-75ae-4054-8143-805d2db6e77c" />



## Code Structure
Code structure is similar to original work by Danijar Hafner in Tensorflow

`dreamer.py`  - main function for training and evaluating dreamer agent

`utils.py`    - Logger, miscallaneous utility functions

`models.py`   - All the networks for world model and actor are implemented here

`replay_buffer.py` - Experience buffer for training world model

`env_wrapper.py`  - Gym wrapper for Dm_control suite

All the hyperparameters are listed in main.py and are avaialble as command line args.


