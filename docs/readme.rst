========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|


.. |docs| image:: https://img.shields.io/readthedocs/laser-polio.svg
    :alt: Documentation Status
    :target: https://docs.idmod.org/projects/laser-polio/en/latest/

.. |github-actions| image:: https://github.com/InstituteforDiseaseModeling/laser-polio/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/InstituteforDiseaseModeling/laser-polio/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser-polio/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser-polio

.. |version| image:: https://img.shields.io/pypi/v/laser-polio.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/laser-polio

.. |wheel| image:: https://img.shields.io/pypi/wheel/laser-polio.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/laser-polio

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/laser-polio.svg
    :alt: Supported versions
    :target: https://pypi.org/project/laser-polio

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/laser-polio.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/laser-polio

.. |commits-since| image:: https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-polio/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/InstituteforDiseaseModeling/laser-polio/compare/v0.0.0...main



.. end-badges

PHASER - Polio Huge Agent Simulation for ERadication

* Free software: MIT license

Installation
============

::

    pip install laser-polio

You can also install the in-development version with::

    pip install https://github.com/InstituteforDiseaseModeling/laser-polio/archive/main.zip


Documentation
=============


https://docs.idmod.org/projects/laser-polio/en/latest/


Development
===========

We strongly recommend using ``uv`` to create and manage virtual environments, including configuring ``tox`` to use ``uv`` for building test environments. To install ``uv`` run::

    pip install uv

To create a new environment run::

    uv venv --python=3.12 .venv

To activate the environment on Linux or MacOS run::

    source .venv/bin/activate

To activate the environment on Windows run::

    .venv\Scripts\activate.bat

To configure ``tox`` to use ``uv`` run::

    uv tool install tox --with tox-uv

To run all the tests run::

    tox

There are several ``tox`` environments that can be run individually. To see the full list of environments run::

    tox -l

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
