# README

## Overview

This repository contains the code used to generate all the simulation data and parameter matrices deployed on devices as discussed in our paper. If you want to utilize this code, you will need to install all the dependencies listed in the `requirements.txt` file.

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the main program, use the following command:

```bash
python main.py [--argument_name argument_value]
```

### Command Line Arguments

The program accepts the following command-line arguments:

- `image_size`: The compressed size of the input image. Available options are [7, 14, 28].
- `layer_num`: The number of hidden layers. The range is from 1 to 4.
- `train_epoch`: The number of training epochs.
- `hidden_layer_size`: The number of neurons in each hidden layer.
- `distill_epoch`: The number of epochs for knowledge distillation.
- `prune_amount`: The pruning ratio, ranging from 0 to 1.

## Example

To run the program with specific parameters, you might use a command like the following:

```bash
python main.py --image_size 14 --layer_num 4 --train_epoch 20 --hidden_layer_size 10 --distill_epoch 5 --prune_amount 0.5
```

This command sets the input image size to 28x28, uses 3 hidden layers, trains for 50 epochs, has 100 neurons per hidden layer, distills knowledge for 20 epochs, and prunes 50% of the network parameters.

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE.txt)

## Acknowledgments

This work was supported by King Abdullah University of Science and Technology. We would like to thank all the contributors who have invested their time and effort in refining this project.