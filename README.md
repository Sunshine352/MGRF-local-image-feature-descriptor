# MGRF local image feature descriptor

MGRF-local-image-feature-descriptor is open source with a public repository on GitHub. It includes the LMGRF, OMGRF descriptors that are published in [1]. Through cascading LMGRF and OMGRF descriptors, we can obtain the final MGRF descriptor[1].

L/OMGRF descriptor is built by Visual Studio 2015 and depended on OPENCV 3.1.0

You can download the all codes, and respectively create LMGRF and OMGRF Projects based on VS2015.

In LMGRF and OMGRF projects, command-line parameters need to be set as follows:

    -i img1.pgm -f img1.hesaff -o img1.hesaff.LMGRF -Order 6 -nSampling 4 -R 1
    -i img1.pgm -f img1.hesaff -o img1.hesaff.OMGRF -Order 6 -nSampling 4 -R 1
    
Specifically, img1.pgm is the source image, img1.hesaff.xxx is the descriptor file descriptor in corresponding with the detector file img1.hesaff, which is the output file of Hessan-Affine Detector and its format is the same as Oxford affine format, as is shown below:

    1.0
    m
    u1 v1 a1 b1 c1
          :
          :
    um vm am bm cm

### Parameters defining an affine region

    u,v,a,b,c in a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1 with (0,0) at image top left corner
    
## The meaning of other command-line parameters are shown in the code detailly

## If this code is helpful to you, please cite the following paper:

    [1] Z. Sun, F. Zhou and Q. Liao, "A robust feature descriptor based on multiple gradient-related features," 2017 IEEE International   
    Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, 2017, pp. 1408-1412.
