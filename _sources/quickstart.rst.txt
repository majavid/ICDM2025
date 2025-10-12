Quickstart
==========

Installation
------------

.. code-block:: console

   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -e .[dev,docs]

Run demos
---------

.. code-block:: console

   icdm2025-demo fit-on-source --data_dir data
   icdm2025-demo first-order-em --data_dir data
   icdm2025-demo ecme --data_dir data
   icdm2025-demo px-em --data_dir data
   icdm2025-demo kiiveri --data_dir data

Run from config
---------------

.. code-block:: console

   icdm2025-run --config configs/first_order_em.yaml
