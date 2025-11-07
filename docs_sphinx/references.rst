References
==========

Academic Papers
---------------

RS-k-means++
~~~~~~~~~~~~

.. [Shah2025] Shah, P., Agrawal, S., & Jaiswal, R. (2025).
   "A New Rejection Sampling Approach to k-means++ With Improved Trade-Offs."
   *arXiv preprint arXiv:2502.02085*.
   https://arxiv.org/abs/2502.02085

k-means++
~~~~~~~~~

.. [Arthur2007] Arthur, D., & Vassilvitskii, S. (2007).
   "k-means++: The advantages of careful seeding."
   *SODA 2007*.
   https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

AFK-MC²
~~~~~~~

.. [Bachem2016] Bachem, O., Lucic, M., Hassani, H., & Krause, A. (2016).
   "Approximate k-means++ in sublinear time."
   *AAAI Conference on Artificial Intelligence*.
   https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12147

.. [Bachem2017] Bachem, O., Lucic, M., & Krause, A. (2017).
   "Distributed and provably good seedings for k-means in constant rounds."
   *ICML 2017*.
   http://proceedings.mlr.press/v70/bachem17a.html

Fast-LSH k-means++
~~~~~~~~~~~~~~~~~~

.. [CohenAddad2020] Cohen-Addad, V., Lattanzi, S., Mitrović, S., Norouzi-Fard, A.,
   Parotsidis, N., & Tarnawski, J. (2020).
   "Fast and accurate k-means++ via rejection sampling."
   *NeurIPS 2020*.
   https://proceedings.neurips.cc/paper/2020/hash/cc384df68c82c0db6d882eadd6871dc9-Abstract.html

Related Work
------------

Clustering
~~~~~~~~~~

.. [Lloyd1982] Lloyd, S. (1982).
   "Least squares quantization in PCM."
   *IEEE Transactions on Information Theory*, 28(2), 129-137.

.. [MacQueen1967] MacQueen, J. (1967).
   "Some methods for classification and analysis of multivariate observations."
   *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297.

Approximate Nearest Neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [JohnsonDouglass2019] Johnson, J., Douze, M., & Jégou, H. (2019).
   "Billion-scale similarity search with GPUs."
   *IEEE Transactions on Big Data*.
   https://arxiv.org/abs/1702.08734

.. [Malkov2018] Malkov, Y. A., & Yashunin, D. A. (2018).
   "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs."
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

External Resources
------------------

Software
~~~~~~~~

- **FAISS**: https://github.com/facebookresearch/faiss
  Efficient similarity search and clustering library from Meta Research

- **scikit-learn**: https://scikit-learn.org
  Machine learning library providing KMeans implementation

- **pybind11**: https://pybind11.readthedocs.io
  C++/Python binding library

Documentation
~~~~~~~~~~~~~

- **Read the Docs**: https://docs.readthedocs.io
  Documentation hosting platform

- **Sphinx**: https://www.sphinx-doc.org
  Documentation generator

Tutorials
~~~~~~~~~

- scikit-learn k-means tutorial: https://scikit-learn.org/stable/modules/clustering.html#k-means
- FAISS tutorial: https://github.com/facebookresearch/faiss/wiki/Getting-started

Citation
--------

If you use kmeans-seeding in your research, please cite:

.. code-block:: bibtex

   @misc{shah2025rejection,
     title={A New Rejection Sampling Approach to k-means++ With Improved Trade-Offs},
     author={Poojan Shah and Swati Agrawal and Ragesh Jaiswal},
     year={2025},
     eprint={2502.02085},
     archivePrefix={arXiv},
     primaryClass={cs.LG}
   }

For the implementation:

.. code-block:: bibtex

   @software{kmeans_seeding2025,
     title={kmeans-seeding: Fast k-means++ Initialization},
     author={Poojan Shah},
     year={2025},
     url={https://github.com/poojanshah/kmeans-seeding}
   }

Contact
-------

- GitHub: https://github.com/poojanshah/kmeans-seeding
- Issues: https://github.com/poojanshah/kmeans-seeding/issues
- PyPI: https://pypi.org/project/kmeans-seeding/
