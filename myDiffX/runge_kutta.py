from jax import jit, vmap, config
from typing import Callable, Tuple
import jax.numpy as jnp
from jax.typing import DTypeLike
import os

# Disable JAX traceback filtering
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


def rk4_step(f: Callable[[float, jnp.ndarray], jnp.ndarray], t: float, y: jnp.ndarray, h: float) -> jnp.ndarray:
    """
    Performs a single time step of the RK4 method 

    Args:
    f: input function defining the ODEs {dyi/dt = f(t,yi)}i
    t: current time
    y: current state
    h: step size
    
    Returns:
    Updated state after one step
    """
    k1 = f(t,y)
    k2 = f(t + h/2, y+h*k1/2)
    k3 = f(t + h/2, y+h*k2/2)
    k4 = f(t+h, y+h*k3)
    
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def rk4(f: Callable[[float, jnp.ndarray], jnp.ndarray], y0: jnp.ndarray, t_range: Tuple[float, float], num_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Top level module that runs rk4_step for the given number of time steps

    Args:
    f: input fuction defining the system of ODEs
    y0: Initial condition for the system of ODEs
    t_range: time domain for the ODEs in the form [initial, final]
    num_steps: number of rk4 steps to calculate the states 

    Returns:
    Pairs of time point and states
    """
    
    t_start, t_end = t_range
    h = (t_end - t_start)/num_steps
    t_array = jnp.linspace(t_start, t_end, num_steps + 1)
    y_array = jnp.zeros((num_steps + 1, *y0.shape))
    y_array = y_array.at[0].set(y0)
    
    def rk4_step_caller(i,y):
        return rk4_step(f, t_array[i], y, h)
    
    y_array = y_array.at[1:].set(vmap(rk4_step_caller, in_axes=(0,0))(jnp.arange(num_steps), y_array[:-1]))

    return t_array, y_array

