# API reference

## Simulation

```{eval-rst}
.. autofunction:: psdm.simulate

.. autoclass:: psdm.Result
   :members:

The dimensionless groups in ``Result.diagnostics`` are described under *Returns* above.

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

.. autodata:: psdm.fitting.Callback
   :no-value:
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

.. autofunction:: psdm.uncertainty.plan

.. autofunction:: psdm.uncertainty.envelope

.. autoclass:: psdm.uncertainty.Run
   :members:

.. autoclass:: psdm.uncertainty.Plan

.. autofunction:: psdm.fit_surface_diffusivity

.. autoclass:: psdm.DsFitResult

.. autofunction:: psdm.fitting.surface_diffusivity_max_evaluations
```

## Errors

```{eval-rst}
.. automodule:: psdm.errors
   :members:
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
