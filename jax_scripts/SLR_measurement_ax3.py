import jax
def SLR_measurement_ax3(meank,Pk,a,weights,W0,Nx,Nz):
    chol_var_mult=jax.numpy.linalg.cholesky((Nx/(1-W0)*Pk))
    sigma_points=[jax.numpy.zeros(Nx,1),jax.numpy.conjugate(chol_var_mult),jax.numpy.conjugate(-chol_var_mult)]
    sigma_points=repmat(meank,1,len(weights))+sigma_points
    #Transformation of sigma points
    sigma_points_z=a*jax.numpy.power(sigma_points,3)
    #Required moments
    z_pred=sigma_points_z*jax.numpy.conjugate(weights)
    var_pred=jax.numpy.zeros(Nz)
    var_xz=jax.numpy.zeros(Nx,Nz)
    for j in range (1,len(weights)+1):
        sub_z_j=sigma_points_z[:,j]-z_pred
        var_pred=var_pred+weights(j)*(sub_z_j* jax.numpy.conjugate(sub_z_j))
        var_xz=var_xz+weights(j)*(sigma_points[:,j]-meank)*jax.numpy.conjugate(sub_z_j)
    #Statistical linear regression parameters
    A_l=jax.numpy.conjugate(var_xz)/Pk
    b_l=z_pred-A_l*meank
    Omega_l=var_pred-A_l*Pk*jax.numpy.conjugate(A_l)
    return [A_l,b_l,Omega_l]