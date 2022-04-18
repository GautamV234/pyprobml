import jax
Nsteps=50 #Number of time steps
Nmc=1000 #Number of Monte Carlo runs
Nmc_trajectory=50 #Number of Monte Carlo runs per trajectory
N_trajectories=Nmc/Nmc_trajectory #Number of simulated trajectories


x0=5
Q=1
a=1/20 #Parameter multiplying x^3
R=1
P_ini=4

alfa_mod=0.9
beta_mod=10
gamma_mod=8

chol_ini= jax.numpy.conjugate(jax.numpy.linalg.cholesky(P_ini))
chol_Q= jax.numpy.conjugate(jax.numpy.linalg.cholesky(Q))
chol_R=jax.numpy.conjugate(jax.numpy.linalg.cholesky(R))


#Ground truth 
X_multi_series=jax.numpy.zeros(N_trajectories,Nsteps);

for i in range (1,N_trajectories+1):
    X_multi_i=jax.numpy.zeros(1,Nsteps)
    xk=x0+chol_ini*jax.random.normal(1)
    X_multi_i[1]=xk
    for k in range (1,Nsteps):
        xk_pred=alfa_mod*xk+beta_mod*xk/(1+xk^2)+gamma_mod*jax.numpy.cos(1.2*k)+chol_Q*jax.random.normal(1)
        X_multi_i[k+1]=xk_pred
        xk=xk_pred
    X_multi_series[i,:]=X_multi_i

noise_z=jax.random.normal(1,Nsteps*Nmc) #Measurement noise