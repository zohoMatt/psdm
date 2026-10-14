# API reference

## Simulation

```{eval-rst}
.. autofunction:: psdm.simulate

.. autoclass:: psdm.Result
   :members:

.. autoclass:: psdm.Column
   :members:

.. autoclass:: psdm.Compound
   :members:
```

## Fitting

```{eval-rst}
.. autofunction:: psdm.fit_freundlich

.. autoclass:: psdm.FitResult

.. autofunction:: psdm.sum_of_squares
```

## Capacity, uncertainty and diffusivity

```{eval-rst}
.. autofunction:: psdm.estimate_capacity

.. autoclass:: psdm.CapacityEstimate
   :members: k_for

.. autodata:: psdm.capacity.METHODS
   :no-value:

.. autofunction:: psdm.capacity.logistic

.. autofunction:: psdm.model_uncertainty

.. autoclass:: psdm.Uncertainty

.. autodata:: psdm.uncertainty.PARAMETERS
   :no-value:

.. autofunction:: psdm.fit_surface_diffusivity

.. autoclass:: psdm.DsFitResult
```

## Full-scale design

```{eval-rst}
.. automodule:: psdm.fullscale
   :members:
```

## Isotherms

```{eval-rst}
.. automodule:: psdm.isotherms
   :members:
```

## Input files

```{eval-rst}
.. automodule:: psdm.io
   :members: read_excel, Dataset
```

## Model building blocks

```{eval-rst}
.. automodule:: psdm.fouling
   :members:

.. automodule:: psdm.correlations
   :members:

.. automodule:: psdm.properties
   :members:

.. automodule:: psdm.units
   :members:

.. automodule:: psdm.collocation
   :members:

.. automodule:: psdm.plotting
   :members:
```
