Biased Bagging for Unsupervised Domain Adaptation
--------------------------

This is the implementation of the ABiB method for domain adaptation, plus accompanying code to test it on a variety of datasets.

The method itself is implemented in `src/abib/predict_abib.m`.
Example:

    addpath src/abib
    addpath src/evaluation
    data = load_dataset('amazon');
    % Predict labels of 'dvd' target domain using 'books' as source
    % Preprocess data: divide by standard deviation over both domains
    [x_src, x_tgt] = preprocess(data.x{1}, data.y{1}, data.x{2}, 'joint-std');
    y = predict_abib(x_src, data.y{1}, x_tgt);
    mean(y == data.y{2})

The method requires that the MATLAB/octave bindings for liblinear are installed. These can be downloaded from https://www.csie.ntu.edu.tw/~cjlin/liblinear/

As a starting point for the experiments, look at `src/evaluation/run_methods.m`.

If you use this code, please cite

    Biased Bagging for Unsupervised Domain Adaptation
    Twan van Laarhoven, Elena Marchiori
    2017

Additional information can be found at http://twanvl.nl/research/domain-adaptation-2017/
The datasets used in the experiments can also be downloaded there.

Comparison methods
----
The comparison methods can be downloaded from:
* GFK: http://www-scf.usc.edu/~boqinggo/domainadaptation.html
* FLDA: https://github.com/wmkouw/da-fl
* Domain adaptation toolbox: https://github.com/viggin/domain-adaptation-toolbox
* Coral: included in the source code
* Subspace Alignment: included in the source code

License
----
The ABiB method and evaluation code is provided under the MIT license (see file LICENSE).
The comparison methods may be subject to other licenses.
