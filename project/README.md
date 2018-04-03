# Group-invariant scattering for GPUs

This project contains a GPU implementation of the scattering transform package
[ScatNet](http://www.di.ens.fr/data/software/scatnet/).  THIS REPOSITORY IS STILL
UNDER DEVELOPMENT, THE FOLLOWING DESCRIPTION HAS NOT YET BEEN IMPLEMENTED.

### Description
Given an input image, this code will compute the scattering transform.  This
consists of applying a series of convolutions, and outputting the coefficients.
Specifically, the following layers are performed.

##### Layer 1
- Gaussian convolution and log<sub>2</sub> downsampling are applied 4 times
successively, producing 5 output images (including the original image). The
final image is the output of this layer.

##### Layer 2
- Each of the 8 Morlet wavelets are convolved with each of the first 4 images
from the previous layer (the output of the previous layer is not included).
No downsampling is applied.  This produces 32 new images.
- For each of these new images, Gaussian convolution and log<sub>2</sub>
downsampling are applied until the image is of the same size as the output from
layer 1.  Again, all of the intermediate images are saved, and the 32 final
images are the output of this layer.

##### Layer 3
- Each of the 8 Morlet wavelets are convolved with each of the intermediate
images from the previous layer (again, the output is not included).  No
downsampling is applied.  This produces 384 new images.
- For each of these new images, Gaussian convolution and log<sub>2</sub>
downsampling are applied until the image is of the same size as the output from
layer 1.  Since this is the last layer, there is no need to save the intermediate
images.  Simply use the 384 final images as the output of this layer.

### Running the code
To run on CHPC with the default `mountains.ppm` image, execute the following
commands:
```
make
sbatch runscript.sh
```
this will save the result in a file called `result.ppm`

### Results
These results were acquired by performing the scattering transform on the image "mountains.ppm".
Sequential: 3.6781156 seconds
Spatial 2D: 0.0257737 seconds
Spatial 1D: 0.0456419 seconds
Fourier:    ???

### References
S. Mallat, "Group Invariant Scattering," _Communications in Pure and Applied
Mathematics_, vol. 65, pp. 1331-1398, October 2012.

L. Sifre, J. Andén, E. Oyallon, M. Kapoko, V. Lostanlen, "ScatNet",
http://www.di.ens.fr/data/software/scatnet/, November 2013.
