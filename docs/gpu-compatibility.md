# RTX 5080 GPU compatibility investigation

Researched on 2026-10-14. This note separates upstream documentation and local
inspection from an environment that has passed InfernoSaber's runtime checks.
The initial candidate below was subsequently tested successfully. See
[gpu-execution.md](gpu-execution.md) for the verified package set, device checks,
model compatibility correction, and real-song GPU generation. The
[CPU baseline](development.md) remains available separately.

## Initial candidate

Keep Python 3.10.19 and NumPy 1.26.4. Test TensorFlow 2.21.0 with tf-keras 2.21.0,
keras-tcn 3.5.4, and a consistent CUDA 12.8 or 12.9 library set. For a conservative
CUDA 12.8 experiment, published components include runtime 12.8.90, nvcc/nvrtc
12.8.93, cuBLAS 12.8.4.1, and cuDNN 9.8.0.87. The complete resolved NVIDIA
package set still needs a lock after a successful experiment. TensorFlow's
`and-cuda` extra allows newer CUDA 12.x/cuDNN 9.x packages; its lower bounds alone
do not establish RTX 5080 compatibility. TensorFlow 2.21 publishes a Python 3.10
Linux GPU wheel, requires NumPy >=1.26, Keras >=3.12, and h5py >=3.11,<3.15;
tf-keras 2.21 requires TensorFlow >=2.21,<2.22.
[TensorFlow package metadata](https://pypi.org/pypi/tensorflow/2.21.0/json),
[tf-keras package metadata](https://pypi.org/pypi/tf-keras/2.21.0/json),
[NVIDIA cuDNN 9.8 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html).

TensorFlow 2.20.0 with tf-keras 2.20.1 is an alternative if 2.21 introduces a
project-specific regression. It has the same relevant CUDA/cuDNN lower bounds;
changing to 2.20 does not itself establish native Blackwell support.
[TensorFlow 2.20 metadata](https://pypi.org/pypi/tensorflow/2.20.0/json),
[tf-keras 2.20.1 metadata](https://pypi.org/pypi/tf-keras/2.20.1/json).

## Documented GPU constraints

- NVIDIA lists the RTX 5080 as compute capability 12.0. Its cuDNN 9.8 support
  matrix explicitly covers capability 12.0 with CUDA 12.8 and Windows driver
  >=570.65. The inspected host's Windows driver 617.14 exceeds that minimum.
  [GPU list](https://developer.nvidia.com/cuda/gpus),
  [cuDNN matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html).
- TensorFlow 2.21's release build defaults are CUDA 12.5.1, cuDNN 9.3, and targets
  `sm_60,sm_70,sm_80,sm_89,compute_90`. This is evidence for embedded compute-90
  PTX, **not** a native sm_120 wheel. NVIDIA documents PTX forward compatibility;
  inference is that this provides a plausible route to execution on the RTX
  5080, subject to every required kernel and library passing actual checks.
  [TensorFlow 2.21 build configuration](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc),
  [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html).
- WSL uses the Windows driver. Do not install a Linux NVIDIA display driver into
  WSL. Current TensorFlow GPU installation uses `tensorflow[and-cuda]` under
  WSL2; native Windows GPU support stopped at TensorFlow 2.10.
  [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
  [TensorFlow installation](https://www.tensorflow.org/install/pip).

## Preserve the legacy Keras model path

Install tf-keras matching TensorFlow's minor version and set
`TF_USE_LEGACY_KERAS=1` before importing TensorFlow. This redirects
`tensorflow.keras` to legacy Keras 2. **It does not redirect standalone
`import keras` or `from keras...`**, which still select the separately installed
Keras 3 dependency. Project imports must consistently select `tensorflow.keras`
with the environment flag, or explicitly select `tf_keras`. Mixing Keras 2 and
Keras 3 layers is unsupported.
[Keras installation guidance](https://keras.io/getting_started/),
[Keras 3 compatibility guidance](https://keras.io/keras_3/).

Local inspection found standalone Keras imports in `training/helpers.py` and
several training/model-building modules. The published keras-tcn 3.5.4 wheel
uses `tensorflow.keras` and imports TensorFlow Addons lazily only for
`use_weight_norm=True`. Its saved TCN config in the local `fav_15` beat model has
`use_weight_norm=False`; no other model in that five-file set contains a TCN.
Thus avoiding Addons execution is plausible for this set, pending forward-pass
checks. Addons may remain installed to satisfy keras-tcn's package metadata;
that does not imply Addons runtime compatibility with newer TensorFlow.
[keras-tcn 3.5.4 distribution and metadata](https://pypi.org/project/keras-tcn/3.5.4/).

Avoid an unpinned keras-tcn upgrade: the inspected 3.5.6/3.5.7 source removes the
`use_weight_norm` parameter present in these legacy H5 configs, and the published
3.5.7 metadata requires `tensorflow-macos`. TensorFlow Addons 0.23 was built
against TensorFlow 2.15 and its project has ended development; it is not a
documented supported component of a TensorFlow 2.21 runtime.
[keras-tcn 3.5.6 distribution](https://pypi.org/project/keras-tcn/3.5.6/),
[keras-tcn 3.5.7 metadata](https://pypi.org/pypi/keras-tcn/3.5.7/json),
[Addons 0.23 release](https://github.com/tensorflow/addons/releases/tag/v0.23.0),
[Addons project status](https://github.com/tensorflow/addons).

## Required local evidence

Use a separate environment, preserving the CPU baseline. Check dependency
resolution; force actual GPU matrix, convolution, and recurrent operations;
then load and run all five original H5 models through the project's loader.
Confirm device placement and finite results rather than relying only on GPU
enumeration. Finally run the app and generate the existing sample at `fav_15` /
BPS 5 with Beat Saber export disabled. Record cold PTX compilation separately
from subsequent timing. These checks establish the project's inference path;
they do not establish training compatibility.
