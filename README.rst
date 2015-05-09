Introduction
------------
Here, we provide a Python implementation for BH-FDR and HICCUPS, two peak calling algorithms for
Hi-C data, proposed by Rao et al [1]_.

Requirements
------------
Following Python Libraries are required:

a) numpy
b) scipy
c) scikit-learn
d) statsmodels
e) mirnylib

You may need ``conda`` to install the first four::

    $ conda install numpy scipy scikit-learn statsmodels

You should install ``mirnylib`` from the source code. (Please refer to my ``HiC_pipeline``
repository for more details)


Usage
-----
Just type ``python BHFDR.py`` or ``python hiccups.py`` for help information.


Reference
---------
.. [1] Rao SS, Huntley MH, Durand NC et al. A 3D Map of the Human Genome at Kilobase Resolution
      Reveals Principles of Chromatin Looping. Cell, 2014, 159(7):1665-80.
