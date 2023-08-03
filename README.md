
## Library
- Run `pip install -r requirement.txt`

## Dataset: Oxford5k
- Download here: [Link](https://drive.google.com/file/d/1ZKImhtRyfoFdtEI1ScDEzKtlG1Nk-cOv/view?usp=sharing)
- Then Untar the file using script
```
import tarfile
filename = "oxbuild_images.tgz"
tf = tarfile.open(filename)
tf.extractall('oxford5k')
```
## Instruction
- Input: filename
- Example: `bodleian_000132.jpg`. Find it from [images_demo](https://github.com/luanvu2307/diffusion/tree/master/images_demo)
- Output: list of path images
## function demo: [demo.py](https://github.com/luanvu2307/diffusion/blob/master/demo.py)
 + demo(filename): search an image using feature extraction from based diffusion
 + demo_custom(filename): search an image using custom extractor
