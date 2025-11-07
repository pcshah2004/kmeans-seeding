Contributing
============

Thank you for your interest in contributing to kmeans-seeding!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/kmeans-seeding.git
      cd kmeans-seeding

3. Create a development environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

4. Install FAISS (optional but recommended):

   .. code-block:: bash

      conda install -c pytorch faiss-cpu

Development Setup
-----------------

Build the C++ extension:

.. code-block:: bash

   pip install -e .

Run tests:

.. code-block:: bash

   pytest tests/ -v

Code Style
----------

**Python**

- Follow PEP 8
- Use Black for formatting: ``black python/``
- Use type hints where appropriate
- Write NumPy-style docstrings

**C++**

- Follow Google C++ style (mostly)
- Use clang-format if available
- Name member variables with trailing underscore: ``member_``
- Use snake_case for functions: ``compute_distance()``

Testing
-------

- Write tests for new features
- Maintain test coverage above 90%
- Test both with and without FAISS
- Include edge cases (k=1, n=k, etc.)

Example test:

.. code-block:: python

   def test_rskmeans_basic():
       X = np.random.randn(1000, 10)
       centers = rskmeans(X, n_clusters=5, random_state=42)

       assert centers.shape == (5, 10)
       assert not np.any(np.isnan(centers))

Documentation
-------------

- Update docstrings for new functions
- Add examples to documentation
- Update changelog
- Rebuild docs: ``cd docs_sphinx && make html``

Pull Request Process
--------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. Make your changes and commit:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description"

3. Push to your fork:

   .. code-block:: bash

      git push origin feature/my-new-feature

4. Open a pull request on GitHub

**PR Checklist**:

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Code formatted (Black for Python)
- [ ] No decrease in test coverage

Reporting Issues
----------------

When reporting bugs, please include:

- Python version
- kmeans-seeding version
- FAISS version (if applicable)
- Minimal reproducible example
- Error messages and stack traces

Example issue:

::

   **Bug**: rskmeans crashes with FastLSH when k > d

   **Environment**:
   - Python 3.11
   - kmeans-seeding 0.2.1
   - macOS 14

   **Reproducible example**:
   ```python
   import numpy as np
   from kmeans_seeding import rskmeans

   X = np.random.randn(100, 5)
   centers = rskmeans(X, n_clusters=10, index_type='FastLSH')
   # Crashes here
   ```

   **Error**:
   ```
   RuntimeError: ...
   ```

Feature Requests
----------------

We welcome feature requests! Please open an issue with:

- Clear description of the feature
- Use cases and motivation
- Proposed API (if applicable)
- Willingness to contribute (optional)

Code of Conduct
---------------

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on technical merit

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.

Contact
-------

- GitHub Issues: https://github.com/poojanshah/kmeans-seeding/issues
- Email: poojan.shah@example.com (update with real email)

Thank You!
----------

Your contributions make kmeans-seeding better for everyone. Thank you for your time and effort!
