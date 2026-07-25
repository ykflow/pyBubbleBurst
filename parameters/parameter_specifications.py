GAUSSIAN_LOCAL_EXPLOSIONS_SPEC = [

    ("delta", "identity", "discontinuous", "theta_mu", 0.1),
    ("beta", "0_1", "discontinuous", "theta_mu", 0.95),
    ("gamma", "positive", "discontinuous", "theta_mu", 0.7),

    ("omega", "positive", "continuous", "theta_b", 0.2),
    ("alpha", "1_plus", "continuous", "theta_b", 1.03),

    ("c", "identity", "discontinuous", "theta_g", -0.1),

    ("sigma", "positive", "continuous", "theta_d", 1.0),
]


STUDENT_LOCAL_EXPLOSIONS_SPEC = [

    ("delta", "identity", "discontinuous", "theta_mu", 0.1),
    ("beta", "0_1", "discontinuous", "theta_mu", 0.95),
    ("gamma", "positive", "discontinuous", "theta_mu", 0.7),

    ("omega", "positive", "continuous", "theta_b", 0.2),
    ("alpha", "1_plus", "continuous", "theta_b", 1.03),

    ("c", "identity", "discontinuous", "theta_g", -0.1),

    ("sigma", "positive", "continuous", "theta_d", 1.0),
    ("nu", "1_plus", "continuous", "theta_d", 8.0),
]


EGB2_LOCAL_EXPLOSIONS_SPEC = [

    ("delta", "identity", "discontinuous", "theta_mu", 0.1),
    ("beta", "0_1", "discontinuous", "theta_mu", 0.95),
    ("gamma", "positive", "discontinuous", "theta_mu", 0.7),

    ("omega", "positive", "continuous", "theta_b", 0.2),
    ("alpha", "1_plus", "continuous", "theta_b", 1.03),

    ("c", "identity", "discontinuous", "theta_g", -0.1),

    ("sigma", "positive", "continuous", "theta_d", 1.0),
    ("xi", "positive", "continuous", "theta_d", 2.0),
    ("zeta", "positive", "continuous", "theta_d", 2.0),
]