# Attribute Inference

## How to run
`$ python cli.py attribute-inference [OPTIONS] COMMAND [ARGS]...`

* Options:
  * `--help`  Show this message and exit.

* Commands:
  * `pretrained-dummy`  Load trained target and attack model
  * `supply-target`     Supply own target model and train attack model
  * `train-dummy`       Train target and attack model

#### `pretrained-dummy`
* `$ python cli.py attribute-inference pretrained-dummy [OPTIONS]`

  * Load trained target and attack model

* Options:
  * `--help`  Show this message and exit.


#### `train-dummy`
* `$ python cli.py attribute-inference train-dummy [OPTIONS]`

  * Train target and attack model

* Options:
  * `-t`, `--target_epochs` `INTEGER`  Number of training epochs for the target model [default=30]
  * `-a`, `--attack_epochs` `INTEGER`  Number of training epochs for the attack model [default=50]
  * `--help`                      Show this message and exit.


#### `supply-target`

* `$ python cli.py attribute-inference supply-target [OPTIONS]`

  * Supply own target model and train attack model.
  * Specifications for the target Model:
    * The target model should predict gender of a human [0: male, 1:female]
    * The target model should give the following output: `y, x` where y are the two posteriors and y is the last fully connected layer. E.g. `[ 5.0912e-01, -5.4544e-01], [-0.0656,  0.0087, -0.0543,  ...,  0.0134,  0.0608, -0.0347], [-0.0580, -0.0461, -0.0593,  ..., -0.0223, -0.1237,  0.1260]`
* Options:
  * `-c`, `--class_file` `TEXT`        File that holds the target models nn.Module class  [required]
  * `-s`, `--state_path` `TEXT`        Path of the state dictionary  [required]
  * `-d`, `--dimension` `INTEGER`      Flattend dimension of the layer used as attack modelinput   [required]
  * `-a`, `--attack_epochs` `INTEGER`  Number of training epochs for the attack model [default=30]
  * `--help`                       Show this message and exit.