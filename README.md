# Navier_Stokes
A numerical simulator of the Navier Stokes Equations using a modified Euler Method in JAX.

The Navier Stokes Equations are:
  
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

The method for updating velocity U_n and pressure p_n from time t to time t+dt is as followed:

  1: find the intemediate velocity U*:
  
      U* = U_n - dt (U_n • ∇) U_n + dt µ ∆U_n 
  
  2: Update the pressure by solving:
  
      ∆p_(n+1) = (ρ/dt) ∇ • U*
      
  3: Correct for the final velocity: 
  
      U_(n+1) = U* - (dt/ρ) ∇p_(n+1)
        
The boundary conditions for the velocity are Dirichlet Boundaries, meaning that they are known and unchanging. In this, 
these known values are the values specified in the initial conditions.

The boundary conditions for pressure are Neumann boundaries, meaning that the gradient normal to the boundaries are known (0).
This is because if the pressure gradient is positive, pressure could flow out of the boundary and be lost. 

The method is outlined here: http://hplgit.github.io/INF5620/doc/pub/main_ns.pdf

And this implementation has been used for guidance: https://www.youtube.com/watch?v=BQLvNLgMTQE

There is also a model of n body particles moving about in the fluid. The derivation of this is below.
Settin
  Given a position x and velocity v of the particle, Euler's method can be used to update the position of the particle:

    x_(n_1) = x_n + dt v_n.

  Updating the velocity requires the equation for drag force of a particle in a fluid. This is:

    F = 1/2 ρ (vr)^2 C A

  Where:

    F: Drag force,
    vr: velocity relative to the fluid (v_f - v_p),
    C: Drag Coefficient,
    A: Cross sectional area of the particle. 

  Newtons second law states: 

    F = m dv/dt, 

  where m is the mass of the particle. Using Euler's Method on Newton's Second Law and substituting in the value for drag force gives:

    v_(n+1) = V_n + (dt/2 m) (ρ (vr_n)^2 C A)

  Setting D = C A / 2 m, this gives:

    v_(n+1) = v_n + D dt ρ (vr_n)^2

  This equation can be used to update the velocity at each timestep. 
          
