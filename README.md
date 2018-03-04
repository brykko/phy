# phy project

A modified version of [**phy**](https://github.com/kwikteam/phy), which adds support for 2D spatial activation plots.

See also [**my modified version of phy-contrib**](https://github.com/brykko/phy-contrib).


## Installation

1. Make sure that you have [**miniconda**](http://conda.pydata.org/miniconda.html) installed. You can choose the Python 3.6 64-bit version for your operating system (Linux, Windows, or OS X).
2. **Download the [environment file](https://raw.githubusercontent.com/kwikteam/phy/master/installer/environment.yml).**
3. **Open a terminal** (on Windows, `cmd`, not Powershell) in the directory where you saved the file and type:

    ```bash
    conda env create -n phy
    source activate phy  # omit the `source` on Windows
    pip install git+https://github.com/brykko/phy.git
    pip install git+https://github.com/brykko/phy-contrib.git
    ```
4. **Done**! Now, to use phy, you have to first type `source activate phy` in a terminal (omit the `source` on Windows), and then call `phy`.


### Updating the software

To get the latest version of the software, open a terminal and type:

```
source activate phy  # omit the `source` on Windows
pip install git+https://github.com/brykko/phy.git git+https://github.com/brykko/phy-contrib.git --upgrade
```

### Documentation

Please refer to [**the main phy repository**](https://github.com/kwikteam/phy) for documentation.
