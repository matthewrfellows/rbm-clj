# rbm-clj

A Clojure library for basic Exponential Family Harmonium functionality. This includes the Restricted Boltzmann Machine (RBM), which is a Bernoulli-Bernoulli Harmonium.

## Usage

rbm-clj requires core.matrix and vectorz-clj.
See the project.clj file.
[net.mikera/core.matrix "0.20.0"]
[net.mikera/vectorz-clj "0.20.0"]

Incanter, glos, and quil are required for the examples.

Sample usage can be found in examples.clj.

For example, at the repl:

(use 'rbm-clj.examples)

(example0)

The image data needed to run the examples can be found
here: http://yann.lecun.com/exdb/mnist/.

Download and uncompress the file called t10k-images.idx3-ubyte.gz.

In examples.clj, set the path in the def for test-file-name to point to this file.

Some of the examples in examples.clj save data to disk. The output directory for saving must be set appropriately.
See examples.clj.



## License

Copyright © Matthew Fellows and contributors.

Distributed under the Eclipse Public License either version 1.0.
