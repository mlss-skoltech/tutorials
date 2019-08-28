This is the practical component of the [Machine Learning Summer School, Moscow 2019](https://mlss2019.skoltech.ru/) session on kernels, focusing on hypothesis testing with kernel statistics.

The materials here are most recently by
[Dougal Sutherland](http://www.gatsby.ucl.ac.uk/~dougals/)
with consultation from [Arthur Gretton](http://www.gatsby.ucl.ac.uk/~gretton/),
updated from [a previous course](https://github.com/dougalsutherland/ds3-kernels/),
and based in large part on [earlier materials](https://github.com/karlnapf/ds3_kernel_testing)
by [Heiko Strathmann](http://herrstrathmann.de/).

We'll cover, in varying levels of detail, the following topics:

- Two-sample testing with the kernel Maximum Mean Discrepancy (MMD).
  - Basic concepts of hypothesis testing, including permutation tests.
  - Computing kernel values.
  - Estimators for the MMD.
  - Learning an appropriate kernel function.
- Independence testing with the Hilbert-Schmidt Independence Criterion.


## Dependencies

### Colab

This notebook is [available on Google Colab](https://colab.research.google.com/github/dougalsutherland/mlss-testing/blob/built/testing.ipynb). You don't have to set anything up yourself and it runs on cloud resources, so this is probably the easiest option if you trust that your network connection is going to be reasonably reliable. Make a copy to your own Google Drive to save your progress, and to use a GPU, click Runtime -> Change runtime type -> Hardware accelerator -> GPU. Everything you need is already installed on Colab; use a Python 3 notebook.

### Local setup

Run `check_imports.py` to see if everything you need is installed and downloaded. If that works, you're set; otherwise, read on.


#### Files
There are a few Python files and some data files in the repository. By far the easiest thing to do is just put them all in the same directory:

```
git clone https://github.com/dougalsutherland/mlss-testing
```

#### Python version
This notebook requires Python 3.6+. Python 3.0 was released in 2008, and it's time to stop living in the past; most importart Python projects [are dropping support for Python 2 this year](https://python3statement.org/). If you've never used Python 3 before, don't worry! It's almost the same; for the purposes of this notebook, you probably only need to know that you should write `print("hi")` since it's a function call now, and you can write `A @ B` instead of `A.dot(B)`.

#### Python packages

The main thing we use is PyTorch and Jupyter. If you already have those set up, you should be fine; just additionally make sure you also have (with `conda install` or `pip install`) `seaborn`, `tqdm`, and `sckit-learn`. We import everything right at the start, so if that runs you shouldn't hit any surprises later on.

If you don't already have a setup you're happy with, we recommend the `conda` package manager - start by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then you can create an environment with everything you need as:

```bash
conda create --name mlss-testing --override-channels -c pytorch -c defaults --strict-channel-priority python=3 notebook ipywidgets numpy scipy scikit-learn pytorch=1.1 torchvision matplotlib seaborn tqdm
conda activate mlss-testing

git clone https://github.com/dougalsutherland/mlss-testing
cd mlss-testing
python check_imports.py
jupyter notebook
```

(If you have an old conda setup, you can use `source activate` instead of `conda activate`, but it's better to [switch to the new style of activation](https://conda.io/projects/conda/en/latest/release-notes.html#recommended-change-to-enable-conda-in-your-shell). This won't matter for this tutorial, but it's general good practice.)

(You can make your life easier when using jupyter notebooks with multiple kernels by installing `nb_conda_kernels`, but as long as you install and run `jupyter` from inside the env it will also be fine.)


## PyTorch

We're going to use PyTorch in this tutorial, even though we're not doing a ton of "deep learning." (The CPU version will be fine, though a GPU might let you get slightly better performance in some of the "advanced" sections.)

If you haven't used PyTorch before, don't worry! The API is unfortunately a little different from NumPy (and TensorFlow), but it's pretty easy to get used to; you can refer to [a cheat sheet vs NumPy](https://github.com/wkentaro/pytorch-for-numpy-users/blob/master/README.md) as well as the docs: [tensor methods](https://pytorch.org/docs/stable/tensors.html) and [the `torch` namespace](https://pytorch.org/docs/stable/torch.html#torch.eq). Feel free to ask if you have trouble figuring something out.

You can convert a `torch.Tensor` to a `numpy.ndarray` with [`t.numpy()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.numpy), and vice versa with [`torch.as_tensor()`](https://pytorch.org/docs/stable/torch.html#torch.as_tensor). (These share data when possible.) Doing this breaks PyTorch's ability to track gradients through these objects, but it's okay for things we won't need to take derivatives of. If you have a one-element tensor, you can get a regular Python number out of it with [`t.item()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item).
