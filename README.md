# Deep Learning From Scratch code

This repo contains all the code from the book [Deep Learning From Scratch](https://www.amazon.com/Deep-Learning-Scratch-Building-Principles/dp/1492041416), published by O'Reilly in September 2019.

It was mostly for me to keep the code I was writing for the book organized, but my hope is readers can clone this repo and step through the code systematically themselves to better understand the concepts.

## Structure

Each chapter has two notebooks: a `Code` notebook and a `Math` notebook. Each `Code` notebook contains the Python code for corresponding chapter and can be run start to finish to generate the results from the chapters. The `Math` notebooks were just for me to store the LaTeX equations used in the book, taking advantage of Jupyter's LaTeX rendering functionality.

### `lincoln`

In the notebooks in the Chapters 4, 5, and 7 folders, I import classes from `lincoln`, rather than putting those classes in the Jupyter Notebook itself. `lincoln` is not currently a `pip` installable library; th way I'd recommend to be able to `import` it and run these notebooks is to add a line like the following your `.bashrc` file:

```bash
export PYTHONPATH=$PYTHONPATH:/Users/seth/development/DLFS_code/lincoln
```

This will cause Python to search this path for a module called `lincoln` when you run the `import` command (of course, you'll have to replace the path above with the relevant path on your machine once you clone this repo). Then, simply `source` your `.bashrc` file before running the `jupyter notebook` command and you should be good to go.

### Chapter 5: Numpy Convolution Demos

While I don't spend much time delving into the details in the main text of the book, I have implemented the batch, multi-channel convolution operation in pure Numpy (I do describe how to do this and share the code in the book's Appendix). In [this notebook](05_convolutions/Numpy_Convolution_Demos.ipynb), I demonstrate using this operation to train a single layer CNN from scratch in pure Numpy to get over 90% accuracy on MNIST.
