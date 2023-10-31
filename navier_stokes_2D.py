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
from jax import jit, tree_util
from functools import partial

config.update("jax_disable_jit", False)

"""
Create a class to contain our current state. Note that this has to be turned into a jax pytree so that functions 
can be jitted.

Also, the functions to update the state do not work as class methods, and must instead be separate functions, because
if they alter self as class methods they are not "pure" functions which jax requires. 
"""


class State:
    """
    A class to contain the state of a fluid system.
    
    :param X: nxn array containing the X values (horizontal distances) at each point.
    :param Y: nxn array containing the Y values (vertical distances) at each point.
    :param u: nxn array containing the x velocity at each point.
    :param v: nxn array containing the y velocity at each point.
    :param p: nxn array containing the pressure at each point.
    :param u_bound: nxn array representing the boundary conditions that should be applied to the outer edge at each
    point for the values of u.
    :param v_bound: nxn array representing the boundary conditions that should be applied to the outer edge at each
    point for the values of v.
    :param f_x: nxn array containing the external force in the x direction to be applied at each point. Only used if
    not using Boussinesq Approximation (i.e. T=None)
    :param f_y: nxn array containing the external force in the y direction to be applied at each point. Only used if
    not using Boussinesq Approximation (i.e. T=None)
    :param dt: scalar, representing the size of one timestep.
    :param density: scalar, representing the density at each point. If using Boussinesq Approximation, this is taken
    as the constant component of the density.
    :param T: nxn array or None. If nxn array, this represents the temperature at each point
    :param T0: Scalar, representing average temperature at each point.
    :param expansion: Scalar, the coefficient of thermal expansion (alpha) in the Boussinesq Approximation.
    :param g: Scalar, representing the strength of gravity.
    :param conductivity: Scalar, thermal conductivity (k) in the Boussinesq Approximation.
    :param capacity: Scalar, thermal conductivity (k) in the Boussinesq Approximation.
    :param heat_production: nxn array, representing the amount of heat production at each point.
    :param viscosity: scalar, the viscosity of the fluid.
    :param jacobi_iterations: Scalar, the number of iterations to use to solve for the pressure.
    """

    def __init__(self, X, Y, u, v, p, u_bound, v_bound, 
                 f_x=None, f_y=None, dt=0.00001, density=1.,
                 T=None, T0=0, T_bound=None, expansion=1000000, g=-1000,
                 conductivity=1000, capacity=10, heat_production=None,
                 viscosity=0.1, jacobi_iterations=50):
        """Constructor method."""
        self.X = X
        self.Y = Y
        self.u = u
        self.v = v
        self.p = p
        self.u_bound = u_bound
        self.v_bound = v_bound
        self.f_x = f_x
        self.f_y = f_y
        self.dt = dt
        self.density = density
        self.T = T
        self.T0 = T0
        self.T_bound = T_bound
        self.expansion = expansion
        self.g = g
        self.conductivity = conductivity
        self.capacity = capacity
        self.heat_production = heat_production
        self.viscosity = viscosity
        self.jacobi_iterations = jacobi_iterations
        self.element_length = 1 / (self.u.shape[0] - 1) 
    
    def _tree_flatten(self):
        """
        Define the pytree flatten function. children are variables that can be altered (so treated like jax jitted
        object. aux_data is any non-variable objects, like booleans or things to be iterated over.

        :return: tuple of children and aux_data.
        """
        children = (self.X, self.Y, self.u, self.v, self.p, self.u_bound, self.v_bound, self.f_x, self.f_y, 
                    self.dt, self.density, self.T, self.T0, self.T_bound, self.expansion, self.g, 
                    self.conductivity, self.capacity, self.heat_production, self.viscosity, self.element_length) 
        aux_data = {'jacobi_iterations': self.jacobi_iterations} 
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Define the pytree unflatten function. The @classmethod wrapper must be used so that the input variables are in
        the right place for jax to recognise it when turning this into a pytree.
        :param aux_data: The auxilary data defined in _tree_flatten
        :param children: The children defined in _tree_flatten
        :return: the State with those children and auxilary data.
        """
        return cls(*children, **aux_data)


# This turns the State class into a jax pytree.
tree_util.register_pytree_node(State, State._tree_flatten, State._tree_unflatten)

"""
Another class will be made to contain particle information too. This can inherit from the other class. 
"""


class StateWithParticles(State):
    """
    Define a class to represent the state of the current system. This class includes particle data.

    :param X: nxn array containing the X values (horizontal distances) at each point.
    :param Y: nxn array containing the Y values (vertical distances) at each point.
    :param u: nxn array containing the x velocity at each point.
    :param v: nxn array containing the y velocity at each point.
    :param p: nxn array containing the pressure at each point.
    :param x: List or px1 array containing the x coordinates of the p points.
    :param y: List or px1 array containing the y coordinates of the p points.
    :param dx_dt: List or px1 array containing the x derivatives of the p
    :param dy_dt:
    :param u_bound: nxn array representing the boundary conditions that should be applied to the outer edge at each
    point for the values of u.
    :param v_bound: nxn array representing the boundary conditions that should be applied to the outer edge at each
    point for the values of v.
    :param f_x: nxn array containing the external force in the x direction to be applied at each point. Only used if
    not using Boussinesq Approximation (i.e. T=None)
    :param f_y: nxn array containing the external force in the y direction to be applied at each point. Only used if
    not using Boussinesq Approximation (i.e. T=None)
    :param drag_constant: Scalar, the drag constant for the particles.
    :param dt: scalar, representing the size of one timestep.
    :param density: scalar, representing the density at each point. If using Boussinesq Approximation, this is taken
    as the constant component of the density.
    :param T: nxn array or None. If nxn array, this represents the temperature at each point
    :param T0: Scalar, representing average temperature at each point.
    :param expansion: Scalar, the coefficient of thermal expansion (alpha) in the Boussinesq Approximation.
    :param g: Scalar, representing the strength of gravity.
    :param conductivity: Scalar, thermal conductivity (k) in the Boussinesq Approximation.
    :param capacity: Scalar, thermal conductivity (k) in the Boussinesq Approximation.
    :param heat_production: nxn array, representing the amount of heat production at each point.
    :param viscosity: scalar, the viscosity of the fluid.
    :param jacobi_iterations: Scalar, the number of iterations to use to solve for the pressure.
    :param bounce: Bool, whether particles should bounce when they hit a boundary.
    """
    def __init__(self, X, Y, u, v, p, x, y, dx_dt, dy_dt, u_bound, v_bound, 
                 f_x=None, f_y=None, drag_constant=10, dt=0.00001, density=1., 
                 T=None, T0=None, T_bound=None, expansion=1, g=9.81,
                conductivity=1, capacity=1, heat_production=1,
                 viscosity=0.1, jacobi_iterations=50, bounce=True):
        """Constructor Method."""
        State.__init__(self, X, Y, u, v, p, u_bound, v_bound, f_x, f_y, dt, 
                       density, T, T0, T_bound, expansion, g, 
                       conductivity, capacity, heat_production,
                       viscosity, jacobi_iterations)
        self.x = x
        self.y = y
        self.dx_dt = dx_dt
        self.dy_dt = dy_dt
        self.drag_constant = drag_constant
        self.bounce = bounce
     
    def _tree_flatten(self):
        """
        Define the pytree flatten function. children are variables that can be altered (so treated like jax jitted
        object. aux_data is any non-variable objects, like booleans or things to be iterated over.

        :return: tuple of children and aux_data.
        """
        children = (self.X, self.Y, self.u, self.v, self.p, self.x, self.y, self.dx_dt, self.dy_dt, self.u_bound, self.v_bound, 
                    self.f_x, self.f_y, self.drag_constant, self.dt, self.density, self.T, self.T0, self.T_bound, self.expansion, 
                    self.g, self.conductivity, self.capacity, self.heat_production, self.viscosity) 
        
        aux_data = {'jacobi_iterations': self.jacobi_iterations, 'bounce': self.bounce}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Define the pytree unflatten function. The @classmethod wrapper must be used so that the input variables are in
        the right place for jax to recognise it when turning this into a pytree.

        :param aux_data: The auxilary data defined in _tree_flatten
        :param children: The children defined in _tree_flatten
        :return: the State with those children and auxilary data.
        """
        return cls(*children, **aux_data)


# This turns the StateWithParticles class into a jax pytree.
tree_util.register_pytree_node(StateWithParticles,
                               StateWithParticles._tree_flatten,
                               StateWithParticles._tree_unflatten)


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
def u_intermediate(state):
    """
    Calculates the intermediate velocities in the x direction.

    :param state: State or StateWithParticles class.
    :return: State with state.u updated to the intermediate value.
    """
    if state.T is not None:
        state.u = (state.u - state.dt * (jnp.multiply(state.u, d_dx(state.u, state.element_length)) +
                   jnp.multiply(state.v, d_dy(state.u, state.element_length))) +
                   state.dt * (state.viscosity * laplacian(state.u, state.element_length) +
                   state.g * (d_dx(state.Y, state.element_length) + d_dy(state.Y, state.element_length)) -
                   state.g * state.expansion * (state.T - state.T0)))
    elif state.f_x is None:
        state.u = (state.u - state.dt * (jnp.multiply(state.u, d_dx(state.u, state.element_length)) +
                   jnp.multiply(state.v, d_dy(state.u, state.element_length))) +
                   state.dt * state.viscosity * laplacian(state.u, state.element_length))
    else:
        state.u = (state.u - state.dt * (jnp.multiply(state.u, d_dx(state.u, state.element_length)) +
                   jnp.multiply(state.v, d_dy(state.u, state.element_length))) +
                   state.dt * (state.viscosity * laplacian(state.u, state.element_length) + state.f_x))
    return state


@jit
def v_intermediate(state):
    """
    Calculates the intermediate velocities in the y direction.

    :param state: State or StateWithParticles class.
    :return: State with state.v updated to the intermediate value.
    """
    if state.T is not None:
        state.v = (state.v - state.dt * (jnp.multiply(state.u, d_dx(state.v, state.element_length)) +
        jnp.multiply(state.v, d_dy(state.v, state.element_length))) +
        state.dt * (state.viscosity * laplacian(state.v, state.element_length) + 
                    state.g * (d_dx(state.Y, state.element_length) + d_dy(state.Y, state.element_length)) - 
                    state.g * state.expansion * (state.T - state.T0)))
    elif state.f_y is None:
        state.v = (state.v - state.dt * (jnp.multiply(state.u, d_dx(state.v, state.element_length)) +
                jnp.multiply(state.v, d_dy(state.v, state.element_length))) +
                state.dt * state.viscosity * laplacian(state.v, state.element_length))
    else:
        state.v = (state.v - state.dt * (jnp.multiply(state.u, d_dx(state.v, state.element_length)) +
                jnp.multiply(state.v, d_dy(state.v, state.element_length))) +
                state.dt * (state.viscosity * laplacian(state.v, state.element_length) + state.f_y))
    return state


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


@jit
def p_update(state):
    """
    Update the pressure to the next timestep using the Jacobi Iterative Procedure.

    :param state: State or StateWithParticles class.
    :return: State with state.p updated to next timestep
    """
    
    # Define the right hand side of the pressure equation.
    rhs = state.density / state.dt * (d_dx(state.u, state.element_length) + d_dy(state.v, state.element_length))

    for i in range(state.jacobi_iterations):
        p_next = jnp.zeros_like(state.p)
        p_next = p_next.at[1:-1, 1:-1].set(1/4 * (
            state.p[1:-1, 0:-2]
            +
            state.p[0:-2, 1:-1]
            +
            state.p[1:-1, 2:]
            +
            state.p[2:, 1:-1]
            -
            state.element_length**2
            *
            rhs[1:-1, 1:-1]))
        
        p_next = p_next.at[:, -1].set(p_next[:, -2])
        p_next = p_next.at[0, :].set(p_next[1, :])
        p_next = p_next.at[:, 0].set(p_next[:, 1])
        p_next = p_next.at[-1, :].set(p_next[-1, :])

        state.p = p_next

    return state


"""Step 3: Correct Velocities.

    The formula to correct the velocities is:
    
        U_(n+1) = U* - (dt/ρ) ∇p_(n+1)
        
    In component form, this is:
    
        u_(n+1) = u* - (dt/ρ) ∂p_(n+1)/∂x
        v_(n+1) = v* - (dt/ρ) ∂p_(n+1)/∂y
"""


@jit
def u_update(state):
    """
    Updates the velocities in the x direction to the next timestep.

    :param state: State or StateWithParticles class.
    :return: State with state.u updated to the next timestep.
    """
    state.u = state.u - (state.dt / state.density) * d_dx(state.p, state.element_length)
    return state


@jit
def v_update(state):
    """
    Updates the velocities in the x direction to the next timestep.

    :param state: State or StateWithParticles class.
    :return: State with state.v updated to the next timestep.
    """
    state.v = state.v - (state.dt / state.density) * d_dy(state.p, state.element_length)
    return state


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

@jit
def update_temperature(state):
    """
    Updates the temperature to the next timestep.

    :param state: State or StateWithParticles class.
    :return: State with state.T updated to the next timestep.
    """
    density_T = state.density - state.expansion * state.density * (state.T - state.T0)
    state.T = state.T + state.dt * (- (state.u * d_dx(state.T, state.element_length)
                                       + state.v * d_dy(state.T, state.element_length))
                                    + state.conductivity/(density_T * state.capacity)
                                    * laplacian(state.T, state.element_length) +
                                    state.heat_production / (density_T * state.capacity))
    
    if state.T_bound is None:
        state.T = state.T.at[:, -1].set(state.T[:, -2])
        state.T = state.T.at[0, :].set(state.T[1, :])
        state.T = state.T.at[:, 0].set(state.T[:, 1])
        state.T = state.T.at[-1, :].set(state.T[-2, :])
    else:
        state.T = impose_boundary(state.T, state.T_bound)
        state.T = state.T.at[:, -1].set(state.T[:, -2])
        state.T = state.T.at[0, :].set(state.T[1, :])
        state.T = state.T.at[:, 0].set(state.T[:, 1])
        
    return state


@jit
def progress_timestep(state):
    """
    Progress the velocities, pressure and temperature forward by one timestep of size dt.

    :param state: State or StateWithParticles class.
    :return: State with u, v, p and T updated to the next timestep.
    """
    
    state = u_intermediate(state)
    state = v_intermediate(state)
    
    state.u = impose_boundary(state.u, state.u_bound)
    state.v = impose_boundary(state.v, state.v_bound)

    state = p_update(state)
    
    state = u_update(state)
    state = v_update(state)
    
    state.u = impose_boundary(state.u, state.u_bound)
    state.v = impose_boundary(state.v, state.v_bound)
    
    if state.T is not None:
        state = update_temperature(state)
    
    return state
    
    
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

@jit
def enforce_lower_boundary(state):
    """
    Ensures particles stay above 0, plus optionally makes them bounce from bottom boundaries.

    :param state: State or StateWithParticles class.
    :return: State, with state.x and state.y bounded below by 0, and if state.bounce==True, state.dx_dt and state.dy_dt
    multiplied by -1 in points where the state.x and state.y values were previously less than 0.
    """
    if state.bounce:
        state.dx_dt = jnp.where(state.x < 0, -state.dx_dt, state.dx_dt)
        state.dy_dt = jnp.where(state.y < 0, -state.dy_dt, state.dy_dt)
    state.x = jnp.where(state.x < 0., 0., state.x)
    state.y = jnp.where(state.y < 0., 0., state.y)
    return state      

@jit
def enforce_upper_boundary(state):
    """
    Ensures particles stay above 0, plus optionally makes them bounce from bottom boundaries.

    :param state: State or StateWithParticles class.
    :return: State, with state.x and state.y bounded above by 1, and if state.bounce==True, state.dx_dt and state.dy_dt
    multiplied by -1 in points where the state.x and state.y values were previously greater than 1.
    """
    if state.bounce:
        state.dx_dt = jnp.where(state.x >= 0.99999, -state.dx_dt, state.dx_dt)
        state.dy_dt = jnp.where(state.y >= 0.99999, -state.dy_dt, state.dy_dt)
    state.x = jnp.where(state.x >= 1., 0.999999, state.x)
    state.y = jnp.where(state.y >= 1., 0.999999, state.y)
    return state


@jit
def progress_timestep_with_particles(state):
    """
    Progress the fluid velocities, pressure and temperature, and particle positions and velocities, forward by one
    timestep of size dt.

    :param state: State or StateWithParticles class.
    :return: State with u, v, p, x, y, dx_dt, dy_dt and T updated to the next timestep.
    """
    # Use Euler's Method to find the next x and y values.
    state.x = state.x + state.dt * state.dx_dt
    state.y = state.y + state.dt * state.dy_dt
    
    # Ensure the particle doesn't go outside the boundaries.
    state = enforce_upper_boundary(state)
    state = enforce_lower_boundary(state)
    
    # Use Euler's Method to find the next x and y values.
    state.x = state.x + state.dt * state.dx_dt
    state.y = state.y + state.dt * state.dy_dt

    # Ensure the particle doesn't go outside the boundaries.
    state = enforce_upper_boundary(state)
    state = enforce_lower_boundary(state)

    # Find the number of points, to be used in the closest_point function.
    num_points = jnp.shape(state.u)[0]
    
    # Find the index x and y values for both particles. 
    x_index = closest_point(state.x, num_points)
    y_index = closest_point(state.y, num_points)
    
    # Find the velocities of the fluid at the location of each particle. 
    fluid_velocities_x = state.u[y_index, x_index]
    fluid_velocities_y = state.v[y_index, x_index]
    
    # Find the relative velocity of the fluid to the particle.
    relative_velocities_x = fluid_velocities_x - state.dx_dt
    relative_velocities_y = fluid_velocities_y - state.dy_dt
    
    # Find the new velocities of the particles.
    state.dx_dt = (state.dx_dt + state.drag_constant * state.dt * state.viscosity *
                jnp.square(relative_velocities_x) * jnp.sign(relative_velocities_x))
    state.dy_dt = (state.dy_dt + state.drag_constant * state.dt * state.viscosity *
                jnp.square(relative_velocities_y) * jnp.sign(relative_velocities_y))
    
    # Ensure the particle doesn't go outside the boundaries.
    state = enforce_upper_boundary(state)
    state = enforce_lower_boundary(state)
    
    # Use the original progress_timestep method to find the next fluid velocities and the next pressure.
    state = progress_timestep(state)
    
    return state


""" Testing the legitimacy of the solution. 

    This section contains the following functions:
        [+] div: returns the divergence at each point in the field f = (u, v):
                ∇•f = du/dx + dv/dy
        [+] kinetic_energy: Returns the kinetic energy of the function, assuming the mass in each square to be 1.
 
"""


def div(state):
    """
    Find the divergence of a 2D vector field.

    :param u: nxn array, Velocity of field in x direction.
    :param v: nxn array, Velocity of field in y direction. 
    :return: nxn array of the divergence of the vector field.
    """
    divergence = jnp.zeros_like(state.u)
    divergence = divergence.at[1:-1, 1:-1].set(d_dx(state.u, state.element_length)[1:-1, 1:-1] - d_dy(state.v, state.element_length)[1:-1, 1:-1])
    return divergence


def kinetic_energy(state):
    """
    Find the kinetic energy of a system, assuming the mass in each square is 1.

    :param state: State class of the system.
    :return: The current kinetic energy of the system. 
    """
    density = state.density - state.expansion * state.density * (state.T - state.T0)
    return 1/2 * jnp.sum(jnp.multiply(density, jnp.square(state.u) + jnp.square(state.v)))
        
    
        