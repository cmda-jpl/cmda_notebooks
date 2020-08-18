# CMDA GUI for Jupyter
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/agoodm/cmda_notebooks/master)

![img](https://puu.sh/G5iCA/89ea6c3097.png)

## JPL Climate Summer School (CCS) JupyterHub server
These notebooks are currently deployed at: 

http://jpl-cmda.org

## Quick Tips for CCS students

### Setting up your workspace
Upon logging into your server, you should find the following two directories: 

![img](https://i.imgur.com/IfMwUwe.png)  

**cmda_notebooks** contains CMDA python library and associated notebooks, while **ccs2020** is a copy (created the first time you log in) that is intended to serve as your workspace. Restarting your server will always force a resync of the cmda_notebooks directory, which can be useful for applying bug fixes. **To ensure that your work is not overwritten upon restarts, please save all your work in the ccs2020 directory only.**

### Workspace Overview 

![img](https://i.imgur.com/XsPdZt3.png)

- `cmda.py` and `fillna.py` contains the main library code. Do not edit these!
- `datasets.csv` and `variables.csv` contain a variable and dataset table that you can open to see the list of available datasets and variables.
- `cmda.ipynb` is a template notebook which provides a brief tutorial for using the CMDA GUI in Jupyter.

### Restarting your Server

Restarting your server manually may be necessary to apply updates. To restart, go to **File > Hub Control Panel** and press the "Stop My Server" button twice, then press "Start My Server".

### Restarting your Kernel

If the CMDA GUI is not responsive in the notebook, restart your kernel by going to **Kernel > Restart Kernel**. This can be necessary if you are idle for too long or you have internet connectivity issues.

### Additional Questions?

Contact Alex Goodman (alexander.goodman@jpl.nasa.gov)