"""2D NAVIER STOKES SOLVER

    The functions below are for a 2D Navier Stokes solver, using a modified Euler Method. The Navier Stokes Equations
    are:
    
        du/dt + (U • ∇)U = - 1/ρ + ∇p + µ ∆U + F
        ∇ • u = 0, 
        
    where:  
    
        ρ is density (scalar),

        U = (u, v) is velocity (vector),

        p is pressure (scalar),

        µ is kinematic viscosity (scalar),

        F is the sum of all external forces (vector).
    
    The density and viscosity of the fluid are assumed to be unchanging and known.
    
    The method for updating velocity U_n and pressure p_n from time t to time t+dt is as followed (link to derivation in
    README):
    
        1: find the intermediate velocity U*:

            U* = U_n - dt (U_n • ∇) U_n + dt (µ ∆U_n + F)
        
        2: Update the pressure by solving:
        
            ∆p_(n+1) = (ρ/dt) ∇ • U*
            
        3: Correct for the final velocity: 
        
            U_(n+1) = U* - (dt/ρ) ∇p_(n+1)
            
    The boundary conditions for the velocity are Dirichlet Boundaries, meaning that they are known and unchanging.
    In this, these known values are the values specified in the initial values.
    
    The boundary conditions for pressure are Neumann boundaries, meaning that the gradient normal to the boundaries are
    known (0). This is because if the pressure gradient is positive, pressure could flow out of the boundary and be
    lost.
            
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
        
    Note that in the code below, the indexing is the backwards compared to what is written above. This is because numpy 
    (and by extension jax numpy) indexes rows first. """


@jit
def d_dx(f, element_length):
    """
    Takes nxm array f and returns the 2 points finite difference derivative in the x direction, with
    border derivatives as 0.

    :param f: nxm array
    :param element_length: Distance between points
    :return: nxm array of df/dx at each point
    """
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * element_length))
    return diff


@jit
def d_dy(f, element_length):
    """
    Takes nxm array f and returns the 2 points finite difference derivative in the y direction, with
    border derivatives as 0.

    :param f: nxm array
    :param element_length: Distance between points
    :return: nxm array of df/dy at each point
    """
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * element_length))
    return diff


@jit
def d2_dx2(f, element_length):
    """
    Takes nxm array f and returns the 3 points finite difference second derivative in the x direction,
    with border derivatives as 0.

    :param f: nxm array
    :param element_length: Distance between points
    :return: nxm array of d^2f/dx^2 at each point
    """
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2]) / (element_length**2))
    return diff


@jit
def d2_dy2(f, element_length):
    """
    Takes nxm array f and returns the 3 points finite difference second derivative in the y direction,
    with border derivatives as 0.

    :param f: nxm array.
    :param element_length: Distance between points
    :return: nxm array of d^2f/dy^2 at each point.
    """
    diff = jnp.zeros_like(f)
    diff = diff.at[1:-1, 1:-1].set((f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1]) / (element_length**2))
    return diff


@jit
def laplacian(f, element_length):
    """
    Takes nxm array f and returns the 5 points finite difference laplacian, with border derivatives as 0.

    :param f: nxm array
    :param element_length: Distance between points
    :return: nxm array of ∆f at each point
    """
    return d2_dx2(f, element_length) + d2_dy2(f, element_length)


"""Step 1: Intermediate Velocities.

    The Intermediate Velocity formula is:
    
        U* = U_n - dt (U_n • ∇) U_n + dt (v ∆U_n + F) 
        
    In component form, this is:
    
        u* = u_n - dt(u ∂u/dx + v ∂u/dy) + dt (µ u_n + f_x)
        v* = v_n - dt(u ∂v/dx + v ∂v/dy) + dt (µ v_n + f_y)
        
"""


@jit
def u_intermediate(u, v, element_length, f_x=None, dt=0.00001, viscosity=0.1):
    """
    Calculates the intermediate velocities in the x direction.

    :param u: nxm array of x component of velocities at each point
    :param v: nxm array of y component of velocities at each point
    :param element_length: Distance between points
    :param f_x: nxm array of x component of force at each point
    :param dt: size of time steps
    :param viscosity: viscosity of the fluid
    :return: intermediate x component of velocity for next timestep
    """
    if f_x is None:
        return (u - dt * (jnp.multiply(u, d_dx(u, element_length)) +
                jnp.multiply(v, d_dy(u, element_length))) +
                dt * viscosity * laplacian(u, element_length))
    else:
        return (u - dt * (jnp.multiply(u, d_dx(u, element_length)) +
                jnp.multiply(v, d_dy(u, element_length))) +
                dt * (viscosity * laplacian(u, element_length) + f_x))


@jit
def v_intermediate(u, v, element_length, f_y=None, dt=0.00001, viscosity=0.1):
    """
    Calculates the intermediate velocities in the y direction.

    :param u: nxm array of x component of velocities at each point
    :param v: nxm array of y component of velocities at each point
    :param element_length: Distance between points
    :param f_y: nxm array of y component of force at each point
    :param dt: size of time steps
    :param viscosity: viscosity of the fluid
    :return: intermediate x component of velocity for next timestep
    """
    if f_y is None:
        return (v - dt * (jnp.multiply(u, d_dx(v, element_length)) +
                jnp.multiply(v, d_dy(v, element_length))) +
                dt * viscosity * laplacian(v, element_length))
    else:
        return (v - dt * (jnp.multiply(u, d_dx(v, element_length)) +
                jnp.multiply(v, d_dy(v, element_length))) +
                dt * (viscosity * laplacian(v, element_length) + f_y))


"""Step 2: Update Pressure. 

    The pressure equation is: 

        ∆p_(n+1) = (ρ/dt) ∇ • U*
    
    This can be solved using a Jacobi method. Using the 5 point estimate for the Laplacian above, the pressure equation
    becomes: 
    
        ∆p(i, j) = (1/4) [p(i+1, j) + p(i-1, j) + p(i, j+1) + p(i, j-1) - 4 ε**2 (ρ/dt) ∇ • U*(i, j)]
        
    This can be calculated iteratively to reach an estimate for the pressure at each timestep. 
    
    To ensure that the Neumann boundaries are satisfied, after each Jacobi Iteration, the value of the boundaries are 
    set to be equal to the nearest interior point to them. 
        
    Note that this is partially jitted, so that the jacobi_iterations input will be inputted as an int and 
    not a jax tracer object, thus allowing it to be iterated over. 
"""


@partial(jit, static_argnames=['jacobi_iterations'])
def p_update(u, v, p_prev, element_length, dt=0.00001, density=1., jacobi_iterations=50):
    """
    Update the pressure to the next timestep using the Jacobi Iterative Procedure.

    :param u: nxm array of x component of intermediate velocities at each point
    :param v: nxm array of y component of intermediate velocities at each point
    :param p_prev: nxm array of pressure at each point
    :param element_length: Distance between points
    :param dt: size of time steps
    :param density: density of the fluid
    :param jacobi_iterations: number of times to run the jacobi iterator to update the pressure
    :return: updated pressure for next timestep
    """
    
    # Define the right hand side of the pressure equation.
    rhs = density / dt * (d_dx(u, element_length) + d_dy(v, element_length))

    for i in range(jacobi_iterations):
        p_next = jnp.zeros_like(p_prev)
        p_next = p_next.at[1:-1, 1:-1].set(1/4 * (
            p_prev[1:-1, 0:-2]
            +
            p_prev[0:-2, 1:-1]
            +
            p_prev[1:-1, 2:]
            +
            p_prev[2:, 1:-1]
            -
            element_length**2
            *
            rhs[1:-1, 1:-1]))
        
        p_next = p_next.at[:, -1].set(p_next[:, -2])
        p_next = p_next.at[0, :].set(p_next[1, :])
        p_next = p_next.at[:, 0].set(p_next[:, 1])
        p_next = p_next.at[-1, :].set(p_next[-1, :])

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
    """
    Updates the velocities in the x direction to the next timestep.

    :param u: nxm array of x component of intermediate velocities at each point
    :param p: nxm array of updated pressure at each point
    :param element_length: Distance between points
    :param dt: size of time steps
    :param density: density of the fluid
    :return: nxm array of final x component of velocity for next timestep
    """
    return u - (dt / density) * d_dx(p, element_length)


@jit
def v_update(v, p, element_length, dt=0.00001, density=1.):
    """
    Updates the velocities in the x direction to the next timestep.

    :param v: nxm array of y component of intermediate velocities at each point
    :param p: nxm array of updated pressure at each point
    :param element_length: Distance between points
    :param dt: size of time steps
    :param density: density of the fluid
    :return: nxm array of final y component of velocity for next timestep
    """
    return v - (dt / density) * d_dy(p, element_length)


"""Impose the Dirichlet boundary conditions for velocity. 

    Imposing the Dirichlet boundary conditions just requires changing the outer perimeter of points to that 
    of the initial values. 
"""


@jit
def impose_boundary(f, f_bound):
    """
    Sets the boundaries of f to the values specified in f_bound.

    :param f: nxm array.
    :param f_bound: nxm array. Only the boundaries of this array are relevant.
    :return: nxm array g, with values from f for interior points and f_bound for exterior points.
    """
    f = f.at[0, :].set(f_bound[0, :])
    f = f.at[:, 0].set(f_bound[:, 0])
    f = f.at[:, -1].set(f_bound[:, -1])
    f = f.at[-1, :].set(f_bound[-1, :])
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
def progress_timestep(u_prev, v_prev, p_prev, u_bound, v_bound, element_length,
                      f_x=None, f_y=None, dt=0.00001, density=1.,
                      viscosity=0.1, jacobi_iterations=50):
    """
    Progress the velocities and pressure forward by one timestep of size dt.

    :param u_prev: nxm array of x components of velocity for current timestep.
    :param v_prev: nxm array of y components of velocity for current timestep.
    :param p_prev: nxm array of pressure for current timestep.
    :param u_bound: nxm array of boundary conditions for x component of velocity.
    :param v_bound: nxm array of boundary conditions for y component of velocity.
    :param element_length: Distance between points.
    :param f_x: nxm array of x component of force at each point.
    :param f_y: nxm array of y component of force at each point.
    :param dt: size of time steps.
    :param density: Density of the fluid.
    :param viscosity: Viscosity of the fluid.
    :param jacobi_iterations: Number of times to run the jacobi iterator to update the pressure.
    :return: x component of velocity, y component of velocity, and pressure for next timestep.
    """
    u_int = u_intermediate(u_prev, v_prev, element_length, dt=dt, viscosity=viscosity, f_x=f_x)
    v_int = v_intermediate(u_prev, v_prev, element_length, dt=dt, viscosity=viscosity, f_y=f_y)
    
    u_int = impose_boundary(u_int, u_bound)
    v_int = impose_boundary(v_int, v_bound)

    p_next = p_update(u_int, v_int, p_prev, element_length, dt=dt, density=density, jacobi_iterations=jacobi_iterations)
    
    u_next = u_update(u_int, p_next, element_length, dt=dt, density=density)
    v_next = v_update(v_int, p_next, element_length, dt=dt, density=density)
    
    u_next = impose_boundary(u_next, u_bound)
    v_next = impose_boundary(v_next, v_bound)
    
    return u_next, v_next, p_next


"""Create a function to progress by one timestep, that also updates n body particle positions. 

    The formulas used to update the position x and velocity v of the particle are:
    
        x_(n+1) = x_n + dt v_n,
        v_(n+1) = v_n + D dt ρ (vr_n)^2
        
    Where D is a constant of drag (D = C A / 2m, where C is the coefficient of drag, A is the cross sectional area of 
    the particle and m is the mass of the particle), and vr = (vf - v) is the velocity of the particle relative to the 
    fluid. The derivation of these equation are in the README. 
    
    The x and y coordinates of the object are continuous whilst the fluid is tracked at specific points. For this
    reason, a function is needed to assign each n body particle to it's nearest point in the fluid. This is done in 
    the closest_point function, by subtracting the coordinate value from each point on the axis and seeing which has
    of these has the smallest absolute values. It is vectorised so that it will work on a vector quantity, not just 
    a scalar. 
    
    The functions enforce_lower_boundary and enforce_upper_boundary must also be made to ensure that the particle stays 
    in [0, 1] x [0, 1], and to negate the velocities if the particle hits the wall.
"""


@partial(jnp.vectorize, excluded=[1])
def closest_point(x, num_points):
    """
    Finds the closest points on a 1D grid of [0,1] to each point in a vector x.

    :param x: Vector of points between 0 and 1.
    :param num_points: Number of points in grid of [0, 1].
    :return: vector of indexes of points in the grid closest to each point in x.
    """
    lin = jnp.linspace(0, 1, num_points)
    x_array = x * jnp.ones_like(lin)

    return jnp.argmin(jnp.abs(x_array - lin))


@partial(jit, static_argnames=['bounce'])
def enforce_lower_boundary(x, dx_dt, bounce=False):
    """
    Ensures particles stay above 0, plus optionally makes them bounce from bottom boundaries.

    :param x: Vector of points between 0 and 1.
    :param dx_dt: Vector of velocities of points in x.
    :param bounce: Bool, whether the particle should bounce when it hits a wall or not.
    :return: x, dx_dt with x values bounded below by 0 and (if bounce = True) dx_dt * -1 where x was <0.
    """
    if not bounce:
        return jnp.where(x < 0., 0., x), dx_dt
    else: 
        return jnp.where(x < 0., 0., x), jnp.where(x < 0, -dx_dt, dx_dt)
        

@partial(jit, static_argnames=['bounce'])
def enforce_upper_boundary(x, dx_dt, bounce=False):
    """
    Ensures particles stay below 1, plus optionally makes them bounce from top boundaries.

    :param x: Vector of points between 0 and 1.
    :param dx_dt: Vector of velocities of points in x.
    :param bounce: Bool, whether the particle should bounce when it hits a wall or not.
    :return: x, dx_dt with x values bounded above by 1 and (if bounce = True) dx_dt * -1 where x was >1.
    """
    if not bounce:
        return jnp.where(x >= 1., 0.999999, x), dx_dt
    else:
        return jnp.where(x >= 1., 0.999999, x), jnp.where(x >= 1., -dx_dt, dx_dt)


@partial(jit, static_argnames=['jacobi_iterations', 'bounce'])
def progress_timestep_with_particles(u_prev, v_prev, p_prev, x_prev, y_prev, dx_dt_prev, dy_dt_prev, 
                                     u_bound, v_bound, element_length, f_x=None, f_y=None, drag_constant=1, 
                                     dt=0.00001, density=1., viscosity=0.1, jacobi_iterations=50, bounce=False):
    """
    Progress the velocities and pressure of the fluid, and positions and velocities of the particles,
    forward by one timestep of size dt.

    :param u_prev: nxm array of x components of velocity for current timestep.
    :param v_prev: nxm array of y components of velocity for current timestep.
    :param x_prev: vector of the x positions of the particles for current timestep.
    :param y_prev: vector of the y positions of the particles for current timestep.
    :param dx_dt_prev: Vector of the x velocities of the particles for current timestep.
    :param dy_dt_prev: Vector of the y velocities of the particles for current timestep.
    :param p_prev: nxm array of pressure for current timestep.
    :param u_bound: nxm array of boundary conditions for x component of velocity.
    :param v_bound: nxm array of boundary conditions for y component of velocity.
    :param element_length: Distance between points.
    :param f_x: nxm array of x component of force at each point.
    :param f_y: nxm array of y component of force at each point.
    :param drag_constant: The drag constant for the particles.
    :param dt: size of time steps.
    :param density: Density of the fluid.
    :param viscosity: Viscosity of the fluid.
    :param jacobi_iterations: Number of times to run the jacobi iterator to update the pressure.
    :param bounce: Bool, saying whether the particles should bounce off walls or not.
    :return: x component of velocity, y component of velocity, pressure, x positions of particles,
    y positions of particles, x velocities of particles and y velocities of particles for next timestep.
    """

    # Use Euler's Method to find the next x and y values.
    x_next = x_prev + dt * dx_dt_prev
    y_next = y_prev + dt * dy_dt_prev
    
    # Ensure the particle doesn't go outside the boundaries.
    x_next, dx_dt_prev = enforce_upper_boundary(x_next, dx_dt_prev, bounce=bounce)
    x_next, dx_dt_prev = enforce_lower_boundary(x_next, dx_dt_prev, bounce=bounce)
    
    y_next, dy_dt_prev = enforce_upper_boundary(y_next, dy_dt_prev, bounce=bounce)
    y_next, dy_dt_prev = enforce_lower_boundary(y_next, dy_dt_prev, bounce=bounce)
    
    # Use Euler's Method to find the next x and y values.
    x_next = x_prev + dt * dx_dt_prev
    y_next = y_prev + dt * dy_dt_prev

    # Ensure the particle doesn't go outside the boundaries.
    x_next, dx_dt_prev = enforce_lower_boundary(x_next, dx_dt_prev, bounce=bounce)
    x_next, dx_dt_prev = enforce_upper_boundary(x_next, dx_dt_prev, bounce=bounce)
    
    y_next, dy_dt_prev = enforce_lower_boundary(y_next, dy_dt_prev, bounce=bounce)
    y_next, dy_dt_prev = enforce_upper_boundary(y_next, dy_dt_prev, bounce=bounce)

    # Find the number of points, to be used in the closest_point function.
    num_points = jnp.shape(u_prev)[0]
    
    # Find the index x and y values for both particles. 
    x_index = closest_point(x_prev, num_points)
    y_index = closest_point(y_prev, num_points)
    
    # Find the velocities of the fluid at the location of each particle. 
    fluid_velocities_x = u_prev[y_index, x_index]
    fluid_velocities_y = v_prev[y_index, x_index]
    
    # Find the relative velocity of the fluid to the particle.
    relative_velocities_x = fluid_velocities_x - dx_dt_prev
    relative_velocities_y = fluid_velocities_y - dy_dt_prev
    
    # Find the new velocities of the particles.
    dx_dt_next = (dx_dt_prev + drag_constant * dt * viscosity *
                  jnp.square(relative_velocities_x) * jnp.sign(relative_velocities_x))
    dy_dt_next = (dy_dt_prev + drag_constant * dt * viscosity *
                  jnp.square(relative_velocities_y) * jnp.sign(relative_velocities_y))
    
    # Ensure the particle doesn't go outside the boundaries.
    x_next, dx_dt_prev = enforce_lower_boundary(x_next, dx_dt_prev, bounce=bounce)
    x_next, dx_dt_prev = enforce_upper_boundary(x_next, dx_dt_prev, bounce=bounce)
    
    y_next, dy_dt_prev = enforce_lower_boundary(y_next, dy_dt_prev, bounce=bounce)
    y_next, dy_dt_prev = enforce_upper_boundary(y_next, dy_dt_prev, bounce=bounce)
    
    # Use the original progress_timestep function to find the next fluid velocities and the next pressure.
    u_next, v_next, p_next = progress_timestep(u_prev, v_prev, p_prev, u_bound, v_bound, element_length, 
                                               f_x=f_x, f_y=f_y, dt=dt, density=density, viscosity=viscosity, 
                                               jacobi_iterations=jacobi_iterations)
    
    return u_next, v_next, p_next, x_next, y_next, dx_dt_next, dy_dt_next


""" Testing the legitimacy of the solution. 

    This section contains the following functions:
        [+] div: returns the divergence at each point in the field f = (u, v):
                ∇•f = du/dx + dv/dy
        [+] kinetic_energy: Returns the kinetic energy of the function, assuming the mass in each square to be 1.
 
"""


def div(u, v):
    divergence = jnp.zeros_like(u)
    divergence[1:-1, 1:-1] = d_dx(u) + d_dy(v)
    return divergence


def kinetic_energy(u, v):
    return jnp.sum(jnp.square(u)) + jnp.sum(jnp.square(v))
