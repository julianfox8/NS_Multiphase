# Numerics

The goal of this implementation is a solver with the ability to produce high-fidelity simulations of two-phase immiscible flows, such as those present in atomization processes. To achieve this we must solve the Navier--Stokes equations:
``math
\frac{\partial \rho \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = \frac{1}{\rho} \nabla P + \mu \nabla^2 \mathbf{u} +f_{body}
''

## Grid Setup

The first decision to be made when implementing a CFD solver is to determine the mesh or grid layout. Where will the different quantities (such as mass and momentum) live with space. The two main options are a staggered or collocated approach. Each of these approaches have their merits and drawbacks. For the purpose of Mist.jl we have chosen to use the collocated apprach where all our quantities are stored at the center of our cells. 

!!! need to add a .png of a collocated grid arrangment!

## Interpolation



## Discretization

### Volume-of-fluid method

### Convective terms

### Viscous terms

### Pressure-projection algorithm

### Body forces