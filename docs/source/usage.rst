=====
Usage
=====

Basic usage
------------

1. Create a folder. (it's recommended to create the folder below the folder examples)
2. Create the yaml configuration file. (The default file name is config.yaml. But you can use the name you prefer. If you do not know how to write this yaml file, you can refer to the template file given in the root of folder examples)
3. To execute the simulation, you should type in lbm. Other arguments 
lbm --help to show how to use lbm.
lbm -c[--config] to specify the config file name. If the file name does not exist, the default file config.yaml or the first yaml file will be used.
lbm -v[--verbose] to specify the log detail. 0=only warn, 1=info mode, 2=debug mode.
lbm --clear to clear the output of simulation.
4. postprocess  
