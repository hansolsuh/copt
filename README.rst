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
    source ./venv/bin/active
    pip3 install /PATH/copt


Running the code
================

To run the code after the installation, the user simply needs to run following file with certain input arguments::

    python3 examples/icml_code/vmatos.py 1 1 1 1

There are total of four numerical experiments, and four input arguments represent binary trigger for the respective experiments. 
If one wants to not run a certain experiment, say third, the modified command would be::

    python3 examples/icml_code/vmatos.py 1 1 0 1
    
This command will generate series of pyplot plots.

