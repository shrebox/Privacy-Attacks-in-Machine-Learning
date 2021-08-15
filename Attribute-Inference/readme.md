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

  * Supply own target model and train attack model

* Options:
  * `-c`, `--class_file` `TEXT`        Path of the models nn.Module class  [required]
  * `-s`, `--state_path` `TEXT`        Path of the state dictionary  [required]
  * `-d`, `--dimension` `INTEGER`      Flattend dimension of the layer used as attack
                               modelinput   [required]
  * `-a`, `--attack_epochs` `INTEGER`  Number of training epochs for the attack model [default=30]
  * `--help`                       Show this message and exit.