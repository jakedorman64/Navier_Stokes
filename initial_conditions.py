"""
This contains a function to create the initial conditions for the 2D Navier Stokes Simulation in navier_stokes_2D.py.
Note that other initial conditions can also be used with this model, this is just an example.
"""

import jax
import time
import jax.numpy as jnp
import numpy as np
from divergence_free_field import k2g_fft, compute_spectrum_map, gaussian_mock
from navier_stokes_2D import StateWithParticles, progress_timestep_with_particles


def initial_conditions(num_points=128, num_particles=10, field_scalar=100000, pressure_scalar=1):
    """
    A function to generate initial conditions for the progress_timestep_with_particles function in navier_stokes_2D.py

    :param num_points: Scalar, how many points the [0,1]x[0,1] grid should be split into.
    :param num_particles: Scalar, how many particles should be simulated.
    :param field_scalar: Scalar to change the average magnitude of the velocity in the initial vector field.
    :param pressure_scalar: Scalar to change the variance of the normally distributed pressure.
    :return: StateWithParticles class.
    """
    
    # Make a meshgrid to create our domain.
    X = jnp.linspace(0, 1, num_points)
    Y = jnp.linspace(0, 1, num_points)
    X, Y = jnp.meshgrid(X, Y)
    
    # Define the initial conditions. The next 4 lines define the divergence free field that will be generated.
    karray = np.arange(num_points)
    pk = np.exp(-karray * karray / 5)
    spectrum_map = compute_spectrum_map(pk, num_points)
    field = gaussian_mock(spectrum_map.flatten(), num_points).T
    
    # Generate the divergence free field. The velocities generated tend to be very small, so multiply the velocities by
    # 10000 so that the velocity of the fluid is great enough to move the particles by a noticeable amount given the
    # size of the timesteps.
    u, v = k2g_fft(field*0, field, dx=1, pad=False)
    u = field_scalar * u
    v = field_scalar * v
    
    # Pressure can be picked from a normal distribution at each point.
    p = pressure_scalar * np.random.normal(u)

    # Generated the initial x and y coordinates of the particle. These can be anywhere in the fluid, so generate these
    # uniformly.
    x = np.random.uniform(0, 1, num_particles)
    y = np.random.uniform(0, 1, num_particles)

    # The initial particle velocities can be set to 0.
    dx_dt = jnp.zeros_like(x)
    dy_dt = jnp.zeros_like(x)

    # Define the boundary conditions.
    u_bound = jnp.zeros_like(u)
    v_bound = jnp.zeros_like(u)
    
    T = 0.001 * np.random.normal(u)
    #T = np.zeros_like(X)
    T0 = 0 
    T_bound = np.zeros_like(X)
    heat_production = np.zeros_like(X)
    return StateWithParticles(X, Y, u, v, p, x, y, dx_dt, dy_dt, u_bound, v_bound, bounce=True, T=T, T0=T0, T_bound=T_bound, heat_production=heat_production)
    
    
def heating_fluctuations(state, heating_scalar, noise_scalar):
    """
    A function to fluctuate the bottom heating boundary of the velocity field.

    :param state: State or StateWithParticles class.
    :param heating_scalar: A scalar to change the average magnitude of the velocity at the bottom boundary due to heat.
    :param noise_scalar: A scalar to change the variance of normally distributed noise applied to the bottom boundary.
    :return: State with state.T_bound updated.
    """
    num_points = state.T_bound.shape[0]
    gaussian_curve = heating_scalar * jnp.exp(- jnp.square(jnp.linspace(-2, 2, num_points)))
    key = jax.random.PRNGKey(int(time.time()))
    state.T_bound = state.T_bound.at[-1, :].set(heating_scalar * gaussian_curve +
                                            noise_scalar * gaussian_curve * jax.random.normal(key=key, shape=jnp.shape(gaussian_curve)))
    
    return state