Description
===========

NNEDI3 is an intra-field only deinterlacer. It takes in a frame, throws away one field, and then interpolates the missing pixels using only information from the kept field. It has same rate and double rate modes, and works with YV12, YUY2, and RGB24 input. NNEDI3 is also very good for enlarging images by powers of 2.

Ported from AviSynth plugin http://bengal.missouri.edu/~kes25c/ and borrowed some codes from https://github.com/dubhater/vapoursynth-nnedi3 & https://forum.doom9.org/showthread.php?t=169766.


Usage
=====

The file `nnedi3_weights.bin` is required. On Windows, it must be located in the same folder as `NNEDI3CL.dll`. Everywhere else it can be located either in the same folder as `libnnedi3cl.so`/`libnnedi3cl.dylib`, or in `$prefix/share/nnedi3/`. The build system installs it at the latter location automatically.

    nnedi3cl.NNEDI3CL(clip, int field[, bint dh=False, bint dw=False, int[] planes, int nsize=6, int nns=1, int qual=1, int etype=0, int device=-1, bint list_device=False, bint info=False])

* clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

* field: Controls the mode of operation (double vs same rate) and which field is kept.
  * 0 = same rate, keep bottom field
  * 1 = same rate, keep top field
  * 2 = double rate (alternates each frame), starts with bottom
  * 3 = double rate (alternates each frame), starts with top

* dh: Doubles the height of the input. Each line of the input is copied to every other line of the output and the missing lines are interpolated. If field=0, the input is copied to the odd lines of the output. If field=1, the input is copied to the even lines of the output. field must be set to either 0 or 1 when using dh=True.

* dw: Doubles the width of the input. It does the same thing as `Transpose().nnedi3(dh=True).Transpose()` but also avoids unnecessary data copies when you scale both dimensions.

* planes: A list of the planes to process. By default all planes are processed.

* nsize: Sets the size of the local neighborhood around each pixel (x_diameter x y_diameter) that is used by the predictor neural network. For image enlargement it is recommended to use 0 or 4. Larger y_diameter settings will result in sharper output. For deinterlacing larger x_diameter settings will allow connecting lines of smaller slope. However, what setting to use really depends on the amount of aliasing (lost information) in the source. If the source was heavily low-pass filtered before interlacing then aliasing will be low and a large x_diameter setting wont be needed, and vice versa.
  * 0 = 8x6
  * 1 = 16x6
  * 2 = 32x6
  * 3 = 48x6
  * 4 = 8x4
  * 5 = 16x4
  * 6 = 32x4

* nns: Sets the number of neurons in the predictor neural network. 0 is fastest. 4 is slowest, but should give the best quality. This is a quality vs speed option; however, differences are usually small. The difference in speed will become larger as `qual` is increased.
  * 0 = 16
  * 1 = 32
  * 2 = 64
  * 3 = 128
  * 4 = 256

* qual: Controls the number of different neural network predictions that are blended together to compute the final output value. Each neural network was trained on a different set of training data. Blending the results of these different networks improves generalization to unseen data. Possible values are 1 or 2. Essentially this is a quality vs speed option. Larger values will result in more processing time, but should give better results. However, the difference is usually pretty small. I would recommend using `qual>1` for things like single image enlargement.

* etype: Controls which set of weights to use in the predictor nn.
  * 0 = weights trained to minimize absolute error
  * 1 = weights trained to minimize squared error

* device: Sets target OpenCL device. Use `list_device` to get the index of the available devices. By default the default device is selected.

* list_device: Whether the devices list is drawn on the frame.

* info: Whether the OpenCL-related info is drawn on the frame.


Compilation
===========

Requires `Boost`.

```
./autogen.sh
./configure
make
```
