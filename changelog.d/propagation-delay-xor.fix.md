Fixed an operator-precedence bug in `propagation_delay_substrate`: unit D now
computes XOR of its two inputs for all previous states (previously 128 of 512
rows were wrong).
