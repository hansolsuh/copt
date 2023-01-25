Anonymized code repository for ICML submission of Variable Metric Three Operator Splitting
==========================================================================================

This is anonymized code for ICML 2023 submission of ``Variable Metric Three Operator Splitting`` (VMTOS) paper. 
It should be noted that the code for VMTOS paper is built on top of copt framework.

Installation
============

If one does not have working copt package installed, the most straightfoward way to install and run the code is using ``pip`` with locaal directory flag::

    pip3 install /PATH/copt

If one already has copt installed, then using virtualenv would be recommended to avoid conflicts::

    cd PROJECTPATH
    virtualenv venv
    source ./venv/bin/activate
    pip3 install numpy==1.23.4
    pip3 install scikit-learn==1.1.3
    pip3 install matplotlib==3.6.2
    pip3 install /PATH/copt

This project uses numpy version of 1.23.4, and sklearn version of 1.1.3.

Running the code
================

To run the code after the installation, the user simply needs to run following file with certain input arguments::

    cd examples/icml_code
    python3 vmatos.py 1 1 1 1

There are total of four numerical experiments, and four input arguments represent binary trigger for the respective experiments. 
If one wants to not run a certain experiment, say third, the modified command would be::

    python3 vmatos.py 1 1 0 1
    
This command will generate series of pyplot plots. Note that the python command needs to be executed inside icml_code folder.

