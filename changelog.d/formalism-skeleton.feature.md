Add ``pyphi.formalism`` package: scaffolding for the IIT 3.0 / IIT 4.0
formalism split. Defines :class:`pyphi.formalism.PhiFormalism`,
:class:`ExactFormalism`, :class:`ApproximateFormalism` Protocols and a
:data:`FORMALISM_REGISTRY` (``Registry[PhiFormalism]``) keyed by string
name. The ``PhiFormalism`` Protocol moved from ``pyphi.protocols`` to
``pyphi.formalism.base`` to live alongside its concrete implementations;
the original location re-exports for back-compat. Concrete formalism
classes (``IIT3Formalism``, ``IIT4_2023Formalism``,
``IIT4_2026Formalism``) and the cut-over of
``Subsystem.{find_mip, sia, evaluate_partition}`` follow in subsequent
commits. Behavior unchanged.
