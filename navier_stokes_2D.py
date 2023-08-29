"""2D NAVIER STOKES SOLVER

    The functions below are for a 2D Navier Stokes solver, using a modified Euler Method. The Navier Stokes Equations are:
    
        du/dt + (U • ∇)U = - 1/ρ + ∇p + µ ∆U + F
        ∇ • u = 0, 
        
    where:  
    
        ρ is density (scalar),

        U = (u, v) is velocity (vector),

        p is pressure (scalar),

        µ is kinematic viscosity (scalar),

        F is the sum of all external forces (vector).
    
    The density and viscosity of the fluid are assumed to be unchanging and known, and the external forces at 0, since this is 
    assuming the forces will only act on the exterior of the box, thus the effect of forces can be incorperated into the boundary
    conditions. 
    
    The method for updating velocity U_n and pressure p_n from time t to time t+dt is as followed (link to derivation in README):
    
        1: find the intemediate velocity U*:

            U* = U_n - dt (U_n • ∇) U_n + dt µ ∆U_n 
        
        2: Update the pressure by solving:
        
            ∆p_(n+1) = (ρ/dt) ∇ • U*
            
        3: Correct for the final velocity: 
        
            U_(n+1) = U* - (dt/ρ) ∇p_(n+1)
            
    The boundary conditions for the velocity are Dirichlet Boundaries, meaning that they are known and unchanging. In this, 
    these known values are the values specified in the initial values.
    
    The boundary conditions for pressure are Neumann boundaries, meaning that the gradient normal to the boundaries are known (0).
    This is because if the pressure gradient is positive, pressure could flow out of the boundary and be lost. 
            
"""

import jax.numpy as jnp
from jax.config import config
from jax import jit
from functools import partial

config.update("jax_disable_jit", False)

"""Derivative Functions.

    The derivatives are defined using a finite difference method. For example, given a grid:
    
        (i-ε, j-ε), (i, j-ε), (i+ε, j-ε)
         (i-ε, j),   (i, j),   (i+ε, j)
        (i-ε, j+ε), (i, j+ε), (i+ε, j+ε)

    the first derivatives are:  
    
        df(i, j)_dx = [f(i+1, j) - f(i-1, j)] / 2ε
        df(i, j)_dy = [f(i, j+1) - f(i, j-1)] / 2ε
    
    The second derivatives are:
        d2f(i, j)_dx2 = [f(i+1, j) + f(i-1, j) - 2f(i, j)] / ε**2
        d2f(i, j)_dy2 = [f(i, j+1) + f(i, j-1) - 2f(i, j)] / ε**2
        
    And the laplacian can be found by summing the second derivatives, though explicitly this gives:
    
        ∆f(i, j) = [f(i+1, j) + f(i-1, j) + f(i, j+1) + f(i, j-1) - 4 f(i, j)] / ε**2
        
    Note that in the code below, the indexing is the backwards compared to what is written above. This is because numpy (and by
    extension jax numpy) indexes rows first. """

@jit
def d_dx(f, element_length):
    """Takes nxm array f and returns the 2 points finite difference derivative in the x direction, with border derivatives as 0."""
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[1:-1, 2:] - f[1:-1, 0:-2] ) / ( 2 * element_length))
    return diff

@jit
def d_dy(f, element_length):
    """Takes nxm array f and returns the 2 points finite difference derivative in the y direction, with border derivatives as 0."""
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[2:, 1:-1  ] - f[0:-2, 1:-1] ) / ( 2 * element_length))
    return diff

@jit
def d2_dx2(f, element_length):
    """Takes nxm array f and returns the 3 points finite difference second derivative in the x direction, with border derivatives as 0."""
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[1:-1, 2:  ] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2] ) / (element_length**2))
    return diff

@jit
def d2_dy2(f, element_length):
    """Takes nxm array f and returns the 3 points finite difference second derivative in the y direction, with border derivatives as 0."""
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[2:, 1:-1  ] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1] ) / (element_length**2))
    return diff

@jit
def laplacian(f, element_length):
    """Takes nxm array f and returns the 5 points finite difference laplacian, with border derivatives as 0."""
    return d2_dx2(f, element_length) + d2_dy2(f, element_length)



"""Step 1: Intermediate Velocities.

    The Intermediate Velocity formula is:
    
        U* = U_n - dt (U_n • ∇) U_n + dt v ∆U_n 
        
    In component form, this is:
    
        u* = u_n - dt(u ∂u/dx + v ∂u/dy) + dt µ u_n
        v* = v_n - dt(u ∂v/dx + v ∂v/dy) + dt µ v_n
        
"""

@jit
def u_intermediate(u, v, element_length, dt=0.00001, viscosity=0.1):
    """Calculates the intermediate velocities in the x direction."""
    return u - dt * (jnp.multiply(u, d_dx(u, element_length)) + jnp.multiply(v, d_dy(u, element_length))) + dt * viscosity * laplacian(u, element_length)

@jit
def v_intermediate(u, v, element_length, dt=0.00001, viscosity=0.1):
    """Calculates the intermediate velocities in the y direction."""
    return v - dt * (jnp.multiply(u, d_dx(v, element_length)) + jnp.multiply(v, d_dy(v, element_length))) + dt * viscosity * laplacian(v, element_length)


"""Step 2: Update Pressure. 

    The pressure equation is: 

        ∆p_(n+1) = (ρ/dt) ∇ • U*
    
    This can be solved using a Jacobi method. Using the 5 point estimate for the Laplacian above, the pressure equation
    becomes: 
    
        ∆p(i, j) = (1/4) [p(i+1, j) + p(i-1, j) + p(i, j+1) + p(i, j-1) - 4 ε**2 (ρ/dt) ∇ • U*(i, j)]
        
    This can be calculated iteratively to reach an estimate for the pressure at each timestep. 
    
    To ensure that the Neumann boundaries are satisfied, after each Jacobi Iteration, the value of the boundaries are set
    to be equal to the nearest interior point to them. 
        
    Note that this is partially jitted, so that the jacobi_iterations input will be inputted as an int and not a jax tracer
    object, thus allowing it to be iterated over. 
"""

@partial(jit, static_argnames=['jacobi_iterations'])
def p_update(u, v, p_prev, element_length, dt=0.00001, density=1., jacobi_iterations=50):
    """Update the pressure to the next timestep using the Jacobi Iterative Procedure."""
    
    # Define the right hand side of the pressure equation.
    rhs = density / dt * (d_dx(u, element_length) + d_dy(v, element_length))

    for i in range(jacobi_iterations):
        p_next = jnp.zeros_like(p_prev)
        p_next= p_next.at[1:-1, 1:-1].set(1/4 * (
            p_prev[1:-1, 0:-2]
            +
            p_prev[0:-2, 1:-1]
            +
            p_prev[1:-1, 2:  ]
            +
            p_prev[2:  , 1:-1]
            -
            element_length**2
            *
            rhs[1:-1, 1:-1]))
        
        p_next = p_next.at[:, -1].set(p_next[:, -2])
        p_next = p_next.at[0,  :].set(p_next[1,  :])
        p_next = p_next.at[:,  0].set(p_next[:,  1])
        p_next = p_next.at[-1, :].set(0.0)

        p_prev = p_next

    return p_next

"""Step 3: Correct Velocities.

    The formula to correct the velocities is:
    
        U_(n+1) = U* - (dt/ρ) ∇p_(n+1)
        
    In component form, this is:
    
        u_(n+1) = u* - (dt/ρ) ∂p_(n+1)/∂x
        v_(n+1) = v* - (dt/ρ) ∂p_(n+1)/∂y
"""

@jit
def u_update(u, p, element_length, dt=0.00001, density=1.):
    """Updates the velocities in the x direction to the next timestep."""
    return (u - (dt / density) * d_dx(p, element_length))

@jit
def v_update(v, p, element_length, dt=0.00001, density=1.):
    """Updates the velocities in the x direction to the next timestep."""
    return (v - (dt / density) * d_dy(p, element_length))


"""Impose the Dirichlet boundary conditions for velocity. 

    Imposing the Dirichlet boundary conditions just requires changing the outer perimeter of points to that 
    of the initial values. 
"""

@jit
def impose_boundary(f, f_init):
    """Sets the boundaries of f to the values specified in f_init."""
    f = f.at[0, :].set(f_init[0, :])
    f = f.at[:, 0].set(f_init[:, 0])
    f = f.at[:, -1].set(f_init[:, -1])
    f = f.at[-1, :].set(f_init[-1, :])
    return f


"""Create a function to progress by 1 timestep. 

    To step forward one timestep, the following must be done:

        1: Calculate the intermediate velocities.
        2: Impose the boundary conditions on the intermediate velocities.
        3: Calculate the pressure. (boundary conditions for pressure are already imposed in the Jacobi procedure.)
        4: Correct the velocities. 
        5: Impose the boundary conditions on the corrected velocities.
"""

@partial(jit, static_argnames=['jacobi_iterations'])
def progress_timestep(u_prev, v_prev, p_prev, u_init, v_init, element_length, dt=0.00001, density=1., viscosity=0.1, jacobi_iterations=50):
    """Progress the velocities and pressure forward by one timestep of size dt."""
    u_int = u_intermediate(u_prev, v_prev, element_length, dt=dt, viscosity=viscosity)
    v_int = v_intermediate(u_prev, v_prev, element_length, dt=dt, viscosity=viscosity)
    
    u_int = impose_boundary(u_int, u_init)
    v_int = impose_boundary(v_int, v_init)

    p_next = p_update(u_int, v_int, p_prev, element_length, dt=dt, density=density, jacobi_iterations=jacobi_iterations)
    
    u_next = u_update(u_int, p_next, element_length, dt=dt, density=density)
    v_next = v_update(v_int, p_next, element_length, dt=dt, density=density)
    
    u_next = impose_boundary(u_next, u_init)
    v_next = impose_boundary(v_next, v_init)
    
    return u_next, v_next, p_next