`pyphi.formalism.iit3.sia`, `ces`, `conceptual_info`, and `phi` now raise a
clear `ConfigurationError` when called under a non-IIT-3.0 active formalism;
previously they silently computed hybrid values by dispatching mechanism-level
work through the active formalism.
