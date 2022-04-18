import jax
def SLR_ungm_dynamic (meank,Pk,alfa_mod,beta_mod,gamma_mod,weights,W0,Nx,k):
    Nz=Nx
    # We obtain the sigma points
    chol_var_mult=jax.numpy.linalg.cholesky((Nx/(1-W0)*Pk))
    sigma_points=[jax.numpy.zeros(Nx,1),jax.numpy.conjugate(chol_var_mult),jax.numpy.conjugate(-chol_var_mult)]

    sigma_points=repmat(meank,1,len(weights))+sigma_points


    # Transformed sigma points
    sigma_points_z=alfa_mod*sigma_points+beta_mod* jax.numpy.divide(sigma_points,(1+jax.numpy.power(sigma_points,2))) +gamma_mod*jax.numpy.cos(1.2*k)

    z_pred=sigma_points_z* jax.numpy.conjugate(weights)


    var_pred=jax.numpy.zeros(Nz)
    var_xz=jax.numpy.zeros(Nx,Nz)

    for j in range(1,len(weights)+1):
        sub_z_j=sigma_points_z[:,j]-z_pred
        var_pred=var_pred+weights(j)*(sub_z_j*jax.numpy.conjugate(sub_z_j))
        var_xz=var_xz+weights(j)*(sigma_points[:,j]-meank)*jax.numpy.conjugate(sub_z_j)

    # Statistical linearisaltion
    A_l=jax.numpy.conjugate(var_xz)/Pk  
    b_l=z_pred-A_l*meank
    Omega_l=var_pred-A_l*Pk*jax.numpy.conjugate(A_l)
    return [A_l,b_l,Omega_l]
