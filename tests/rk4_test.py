import pytest
import jax.numpy as jnp
from myDiffX import rk4, rk4_step
from jax import config, jit
from typing import Callable
import os

# Disable JAX traceback filtering
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


@jit
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

def test_rk4_lorenz():
    # Solve the ODE
    params = lorenz_parameters

    
    t, y = rk4(lorenz_system, jnp.array([1.0,1.0,1.0]), (0.0,50.0),5000)
    
    # Test the shape of the output
    assert t.shape == (5001,)
    assert y.shape == (5001, 3)
    
 

    # Test if the time span is correct
    assert t[0] == 0.0 
    assert t[-1] == 50.0
    print(f"States: {y}")
    print(f"Time step: {t}")

    #print(lorenz_system)

if __name__ == "__main__":
    pytest.main([__file__])


