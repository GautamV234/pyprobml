# randn('seed',9)
# rand('seed',9)

import jax
import Scenario_ungm_trajectories as SUT
import time
import matplotlib.pyplot as plt
import linear_kf_full
import linear_kf_update
import linear_rts_smoother
import SLR_measurement_ax3
import SLR_ungm_dynamic

# Parameters of the sigma-point method (unscented transform)
Nx = 1  # Dimension of the state
W0 = 1/3  # Weight of sigma-point at the mean
Nz = 1  # Dimension of the measurement

Wn = (1-W0)/(2*Nx)
# Sigma-points weights (according to the unscented transform)
weights = [W0, Wn*jax.numpy.ones(1, 2*Nx)]
square_error_t_tot = jax.numpy.zeros(1, SUT.Nsteps)

nees_t_tot = jax.numpy.zeros(1, SUT.Nsteps)
rms_t_series = jax.numpy.zeros(1, SUT.Nmc)
square_error_t_series = jax.numpy.zeros(SUT.Nmc, SUT.Nsteps)

square_error_t_tot_smoothing = jax.numpy.zeros(1, SUT.Nsteps)


# Nit_s-> Number of iterations in smoothing
Nit_s = 10
# Number of iterations in filtering (Nit_iplf=1 is similar to the UKF)
Nit_iplf = 1

square_error_t_tot_smoothing_series = jax.numpy.zeros(
    SUT.Nmc, SUT.Nsteps, Nit_s+1)  # For filtering and the N_it_s iterations
# Negative log-likelihood (See Marc Deisenroth PhD thesis)
NLL_smoothing_series = jax.numpy.zeros(SUT.Nmc, SUT.Nsteps, Nit_s+1)
Nees_smoothing_series = jax.numpy.zeros(SUT.Nmc, SUT.Nsteps, Nit_s+1)  # NEES
cte_NLL = Nx/2*jax.numpy.log(2*jax.numpy.pi)  # Constant for NLL

mean_ini = SUT.x0


# randn('seed',9)
# rand('seed',9)


# Number of Monte Carlo runs

for i in range(1, SUT.Nmc+1):
    tic = time.process_time()
    n_trajectory = jax.numpy.fix((i-1)/SUT.Nmc_trajectory)+1
    X_multi = SUT.X_multi_series[n_trajectory, :]
    meank = mean_ini
    Pk = SUT.P_ini

    square_error_t = jax.numpy.zeros(1, SUT.Nsteps)
    square_error_t_smoothing = jax.numpy.zeros(1, SUT.Nsteps)

    nees_t = jax.numpy.zeros(1, SUT.Nsteps)

    meank_t = jax.numpy.zeros(Nx, SUT.Nsteps)
    Pk_t = jax.numpy.zeros(Nx, Nx, SUT.Nsteps)

    # SLR parameters for dynamics
    A_dyn = jax.numpy.zeros(Nx, Nx, SUT.Nsteps)
    b_dyn = jax.numpy.zeros(Nx, SUT.Nsteps)
    Omega_dyn = jax.numpy.zeros(Nx, Nx, SUT.Nsteps)

    # SLR parameters for measurements
    A_m = jax.numpy.zeros(Nz, Nx, SUT.Nsteps)
    b_m = jax.numpy.zeros(Nz, SUT.Nsteps)
    Omega_m = jax.numpy.zeros(Nz, Nz, SUT.Nsteps)

    # Generation of measurements
    z_real_t = jax.numpy.zeros(Nz, SUT.Nsteps)
    for k in range(1, SUT.Nsteps+1):
        state_k = X_multi[:, k]
        noise = SUT.chol_R * SUT.noise_z[:, k+SUT.Nsteps*(i-1)]
        z_real = SUT.a*state_k ^ 3+noise
        z_real_t[:, k] = z_real

    # Filtering

    for k in range(1, SUT.Nsteps+1):
        pos_x = X_multi(1, k)
        z_real = z_real_t[:, k]

        # Calculate iterated SLR
        meank_j = meank
        Pk_j = Pk

        for p in range(1, Nit_iplf+1):
            # SLR of function a*x^3 w.r.t. current approximation of the
            # posterior, with mean meank_j and covariance Pk_j
            [A_l, b_l, Omega_l] = SLR_measurement_ax3(
                meank_j, Pk_j, SUT.a, weights, W0, Nx, Nz)

            # KF update for the linearised model
            [mean_ukf_act, var_ukf_act] = linear_kf_update(
                meank, Pk, A_l, b_l, Omega_l, SUT.R, z_real)

            meank_j = mean_ukf_act
            Pk_j = var_ukf_act

        A_m[:, :, k] = A_l
        b_m[:, k] = b_l
        Omega_m[:, :, k] = Omega_l

        meank_t[:, k] = mean_ukf_act
        Pk_t[:, :, k] = var_ukf_act

        square_error_t[k] = (mean_ukf_act-pos_x) ^ 2
        pos_error = mean_ukf_act-pos_x
        var_pos_act = var_ukf_act(1)
        # nees_t(k)=state_error'*inv(var_mp_tukf_act)*state_error;
        nees_t[k] = jax.numpy.conjugate(pos_error)/var_pos_act*pos_error

        square_error_t_tot_smoothing_series[i, k, 1] = square_error_t(k)
        NLL_smoothing_series[i, k, 1] = 1/2*jax.numpy.log(
            var_ukf_act(1))+1/2*square_error_t(k)/var_ukf_act(1)+cte_NLL
        Nees_smoothing_series[i, k, 1] = square_error_t(k)/var_ukf_act(1)

        # Prediction
        [A_dyn_k, b_dyn_k, Omega_dyn_k] = SLR_ungm_dynamic(
            mean_ukf_act, var_ukf_act, SUT.alfa_mod, SUT.beta_mod, SUT.gamma_mod, weights, W0, Nx, k)

        meank = A_dyn_k*mean_ukf_act+b_dyn_k
        Pk = A_dyn_k*var_ukf_act * \
            jax.numpy.conjugate(A_dyn_k) + Omega_dyn_k+SUT.Q
        Pk = (Pk+jax.numpy.conjugate(Pk))/2

        A_dyn[:, :, k] = A_dyn_k
        b_dyn[:, k] = b_dyn_k
        Omega_dyn[:, :, k] = Omega_dyn_k

    # Smoothing
    [meank_smoothed_t, Pk_smoothed_t] = linear_rts_smoother(
        meank_t, Pk_t, A_dyn, b_dyn, Omega_dyn, SUT.Q)

    for k in range(1, SUT.Nsteps):
        pos_x = X_multi(1, k)
        square_error_t_tot_smoothing_series[i, k, 2] = (
            meank_smoothed_t(1, k)-pos_x) ^ 2
        NLL_smoothing_series[i, k, 2] = 1/2 * jax.numpy.log(Pk_smoothed_t(
            1, 1, k))+1/2*square_error_t_tot_smoothing_series(i, k, 2)/Pk_smoothed_t(1, 1, k)+cte_NLL
        Nees_smoothing_series[i, k, 2] = square_error_t_tot_smoothing_series(
            i, k, 2)/Pk_smoothed_t(1, 1, k)

    for p in range(1, Nit_s):

        # Iterated SLR using the current posterior

        for k in range(1, SUT.Nsteps+1):
            # Generation of sigma points
            meank = meank_smoothed_t[:, k]
            Pk = Pk_smoothed_t[:, :, k]

            # SLR for the measurement
            [A_l, b_l, Omega_l] = SLR_measurement_ax3(
                meank, Pk, SUT.a, weights, W0, Nx, Nz)

            A_m[:, :, k] = A_l
            b_m[:, k] = b_l
            Omega_m[:, :, k] = Omega_l

            # SLR for the dynamics
            [A_dyn_k, b_dyn_k, Omega_dyn_k] = SLR_ungm_dynamic(
                meank, Pk, SUT.alfa_mod, SUT.beta_mod, SUT.gamma_mod, weights, W0, Nx, k)

            A_dyn[:, :, k] = A_dyn_k
            b_dyn[:, k] = b_dyn_k
            Omega_dyn[:, :, k] = Omega_dyn_k

        # Filter with the linearised model

        [meank_t, Pk_t] = linear_kf_full(
            mean_ini, SUT.P_ini, A_m, b_m, Omega_m, A_dyn, b_dyn, Omega_dyn, SUT.R, SUT.Q, z_real_t)

        # Smoother

        [meank_smoothed_t, Pk_smoothed_t] = linear_rts_smoother(
            meank_t, Pk_t, A_dyn, b_dyn, Omega_dyn, SUT.Q)

        for k in range(1, SUT.Nsteps+1):
            pos_x = X_multi(1, k)
            square_error_t_tot_smoothing_series[i, k,
                                                p+2] = (meank_smoothed_t(1, k)-pos_x) ^ 2
            NLL_smoothing_series[i, k, p+2] = 1/2*jax.numpy.log(Pk_smoothed_t(
                1, 1, k))+1/2*square_error_t_tot_smoothing_series(i, k, p+2)/Pk_smoothed_t(1, 1, k)+cte_NLL
            Nees_smoothing_series[i, k, p+2] = square_error_t_tot_smoothing_series(
                i, k, p+2)/Pk_smoothed_t(1, 1, k)

    # Square error calculation

    for k in range(1, SUT.Nsteps+1):
        pos_x = X_multi(1, k)
        square_error_t_smoothing[k] = (meank_smoothed_t(1, k)-pos_x) ^ 2

    square_error_t_tot = square_error_t_tot+square_error_t
    square_error_t_series[i, :] = square_error_t
    square_error_t_tot_smoothing = square_error_t_tot_smoothing+square_error_t_smoothing

    nees_t_tot = nees_t_tot+nees_t
    rms_t_series[i] = jax.numpy.sqrt(sum(square_error_t)/(SUT.Nsteps))
    t = time.process_time()-tic
    print(f"Completed iteration nº {i}, time {t} seconds")

square_error_t_tot = square_error_t_tot/SUT.Nmc
rmse_filtering_tot = jax.numpy.sqrt(sum(square_error_t_tot)/(SUT.Nsteps))

square_error_t_tot_smoothing = square_error_t_tot_smoothing/SUT.Nmc

rmse_tot_smoothing = jax.numpy.sqrt(
    sum(square_error_t_tot_smoothing)/(SUT.Nsteps))

rmse_t_tot_smoothing_series = jax.numpy.sqrt(sum(jax.numpy.squeeze(
    sum(square_error_t_tot_smoothing_series, 1)), 1)/(SUT.Nmc*SUT.Nsteps))


nees_t_tot = nees_t_tot/SUT.Nmc


# Smoothing error for different J
rmse_smoothing_1 = jax.numpy.sqrt(
    sum(square_error_t_tot_smoothing_series[:, :, 2], 1)/SUT.Nmc)
rmse_smoothing_5 = jax.numpy.sqrt(
    sum(square_error_t_tot_smoothing_series[:, :, 6], 1)/SUT.Nmc)
rmse_smoothing_10 = jax.numpy.sqrt(
    sum(square_error_t_tot_smoothing_series[:, :, 11], 1)/SUT.Nmc)


# Output figure (IPLS(i)-J denotes a IPLS with i SLR iterations for the IPLF and
# J SLR smoothing iterations)

fig = plt.figure()
x_axis = jax.numpy.arange(1, SUT.Nsteps)
plt.plot(x_axis, jax.numpy.sqrt(square_error_t_tot), 'b')
plt.plot(x_axis, rmse_smoothing_1, '--r')
plt.plot(x_axis, rmse_smoothing_5, '-.black')
plt.plot(x_axis, rmse_smoothing_10, '-*g', 'Linewidth', 1.3)
# grid on
plt.legend('IPLS(1)-0', 'IPLS(1)-1', 'IPLS(1)-5', 'IPLS(1)-10')
plt.xlabel('Time step')
plt.ylabel('RMS error')
plt.show()


NLL_average_list = jax.numpy.zeros(1, Nit_s)

for i in range(1, Nit_s+1+1):
    NLL = NLL_smoothing_series[:, :, i]
    NLL_average = sum(sum(NLL))/(SUT.Nsteps*SUT.Nmc)
    NLL_average_list[i] = NLL_average
