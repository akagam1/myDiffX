import pytest
import jax.numpy as jnp
from myDiffX.runge_kutta import rk4

def lorenz_system(t: float, state: jnp.ndarray) -> jnp.ndarray:
    """
    The Lorenz system of ODEs.
    """
    x, y, z = state
    sigma, rho, beta = 10.0, 28.0, 8/3
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return jnp.array([dx_dt, dy_dt, dz_dt])

@pytest.fixture
def lorenz_parameters():
    return {
        'y0': jnp.array([1.0, 1.0, 1.0]),
        't_span': (0.0, 50.0),
        'num_steps': 5000
    }

def test_rk4_lorenz(lorenz_parameters):
    # Solve the ODE
    t, y = rk4(lorenz_system, **lorenz_parameters)
    
    # Test the shape of the output
    assert t.shape == (lorenz_parameters['num_steps'] + 1,)
    assert y.shape == (lorenz_parameters['num_steps'] + 1, 3)
    
    # Test if the initial condition is correct
    jnp.testing.assert_allclose(y[0], lorenz_parameters['y0'])
    
    # Test if the time span is correct
    assert t[0] == lorenz_parameters['t_span'][0]
    assert t[-1] == lorenz_parameters['t_span'][1]

    print(f"States: {y}")
    print(f"Time step: {t}")


if __name__ == "__main__":
    pytest.main([__file__])
