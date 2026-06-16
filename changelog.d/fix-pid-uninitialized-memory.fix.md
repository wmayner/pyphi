Fixed ``pointwise_intrinsic_differentiation`` (used by the
``INTRINSIC_DIFFERENTIATION`` measure) calling ``np.log2(p, where=(p > 0))``
without an ``out`` argument. NumPy backs such a call with an *uninitialized*
buffer and only writes the masked-in entries, leaving the entries where
``p == 0`` holding whatever was previously in that memory — which also
triggered a ``UserWarning`` ("``'where'`` used without ``'out'``, expect
unitialized memory in output") on every SIA computation. Because
``intrinsic_differentiation`` negates the result and then keeps entries
``> 0`` before taking the minimum, a small negative garbage value in a
``p == 0`` slot could flip positive, survive the filter, and corrupt the
reported minimum. The call now writes into an explicit zero-initialized
buffer, so masked entries are deterministically ``0.0`` and excluded by the
``> 0`` filter as intended.
