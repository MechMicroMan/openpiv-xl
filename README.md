# OpenPIV-XL

This package is designed to perform digital image correlation on large images (70,000 x 70,000 px has been tested) using chunking and GPU acceleration. Using the GPU, DIC correlation from a sub-window of 2048x2048 to 16x16 with no overlap on a (16,000 x 16,000 px) px image takes around one minute. To compute strain fields and perform advanced analysis of the displacement field, see [DefDAP](https://github.com/MechMicroMan/DefDAP).

The authors aknowledge the authors of [OpenPIV](https://github.com/OpenPIV/openpiv-python), from which this package is derived. OpenPIV should be cited with this [DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4409178).

## Installing

Download the package and run

`pip install -e .`

OR

`pip install git+https://github.com/MechMicroMan/openpiv-python.git`

Note: you will need a Nvidia GPU which supports CUDA 12x to use GPU acceleration

## License and copyright

This package is licenses under the GNU General Public License v3.0
