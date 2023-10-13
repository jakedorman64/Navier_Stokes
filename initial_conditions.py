"""
This contains a function to create the initial conditions for the 2D Navier Stokes Simulation in navier_stokes_2D.py.
Note that other initial conditions can also be used with this model, this is just an example.
"""

import jax
import time
import jax.numpy as jnp
import numpy as np
from divergence_free_field import k2g_fft, compute_spectrum_map, gaussian_mock


def initial_conditions(num_points=128, num_particles=10,
                       gravity_scalar=10, field_scalar=20000,
                       pressure_scalar=1):
    """
    A function to generate initial conditions for the progress_timestep_with_particles function in navier_stokes_2D.py

    :param num_points: Scalar, how many points the [0,1]x[0,1] grid should be split into.
    :param num_particles: Scalar, how many particles should be simulated.
    :param gravity_scalar: Scalar to change the strength of gravitational force.
    :param field_scalar: Scalar to change the average magnitude of the velocity in the initial vector field.
    :param pressure_scalar: Scalar to change the variance of the normally distributed pressure.
    :return: x, y, u_prev, v_prev, p_prev, x_prev, y_prev, dx_dt_prev, dy_dt_prev, u_bound, v_bound, f_x, f_y. The
    initial conditions for progress_timesteps_with_particles.
    """
    
    # Make a meshgrid to create our domain.
    x = jnp.linspace(0, 1, num_points)
    y = jnp.linspace(0, 1, num_points)
    x, y = jnp.meshgrid(x, y)
    
    # Define the initial conditions. The next 4 lines define the divergence free field that will be generated.
    karray = np.arange(num_points)
    pk = np.exp(-karray * karray / 5)
    spectrum_map = compute_spectrum_map(pk, num_points)
    field = gaussian_mock(spectrum_map.flatten(), num_points).T
    
    # Generate the divergence free field. The velocities generated tend to be very small, so multiply the velocities by
    # 10000 so that the velocity of the fluid is great enough to move the particles by a noticeable amount given the
    # size of the timesteps.
    u_prev, v_prev = k2g_fft(field*0, field, dx=1, pad=False)
    u_prev = field_scalar * u_prev
    v_prev = field_scalar * v_prev
    
    # Pressure can be picked from a normal distribution at each point.
    p_prev = pressure_scalar * np.random.normal(u_prev)

    # Generated the initial x and y coordinates of the particle. These can be anywhere in the fluid, so generate these
    # uniformly.
    x_prev = np.random.uniform(0, 1, num_particles)
    y_prev = np.random.uniform(0, 1, num_particles)

    # The initial particle velocities can be set to 0.
    dx_dt_prev = jnp.zeros_like(x_prev)
    dy_dt_prev = jnp.zeros_like(x_prev)

    # Define the boundary conditions.
    u_bound = jnp.zeros_like(x)
    v_bound = jnp.zeros_like(x)

    # The acting force on the fluid will be gravity. This will just be a uniform vector field pointing downwards. 
    f_x = jnp.zeros_like(u_prev)
    # Change 2000 to a different number to change the strength of gravity. 
    f_y = - gravity_scalar * jnp.ones_like(u_prev)
    
    return x, y, u_prev, v_prev, p_prev, x_prev, y_prev, dx_dt_prev, dy_dt_prev, u_bound, v_bound, f_x, f_y


def heating_fluctuations(v_bound, heating_scalar=1, noise_scalar=0.25):
    """
    A function to fluctuate the bottom heating boundary of the velocity field.

    :param v_bound: nxn array, where the bottom values will be altered and returned.
    :param heating_scalar: A scalar to change the average magnitude of the velocity at the bottom boundary due to heat.
    :param noise_scalar: A scalar to change the variance of normally distributed noise applied to the bottom boundary.
    """
    num_points = v_bound.shape[0]
    gaussian_curve = heating_scalar * jnp.exp(- jnp.square(jnp.linspace(-5, 5, num_points)))
    key = jax.random.PRNGKey(int(time.time()))
    return v_bound.at[0, :].set(heating_scalar * gaussian_curve +
                                noise_scalar * jax.random.normal(key=key, shape=jnp.shape(gaussian_curve)))
    