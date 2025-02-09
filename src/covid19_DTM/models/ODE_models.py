# Copyright (c) 2023 by T.W. Alleman BIOMATH, Ghent University. All Rights Reserved.

import numpy as np
from pySODM.models.base import ODE
from covid19_DTM.models.jit_utils import jit_matmul_2D_1D, jit_matmul_3D_2D, jit_matmul_klm_m, jit_matmul_klmn_n, matmul_q_2D

class simple_multivariant_SIR(ODE):
    """
    A minimal example of a SIR compartmental disease model with an implementation of transient multivariant dynamics
    Can be reduced to a single-variant SIR model by setting injection_ratio to zero.

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required

        S : susceptible
        I : infectious
        R : removed
        alpha : fraction of alternative COVID-19 variant

    parameters : dictionary
        containing the values of all parameters (both stratified and not)

        Non-stratified parameters
        -------------------------

        beta : probability of infection when encountering an infected person
        gamma : recovery rate (inverse of duration of infectiousness)
        injection_day : day at which injection_ratio of the new strain is introduced in the population
        injection_ratio : initial fraction of alternative variant

        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """

    # state variables and parameters
    states = ['S', 'I', 'R', 'alpha']
    parameters = ['beta', 'gamma', 'injection_day', 'injection_ratio', 'Nc']
    dimensions = ['age_groups']

    @staticmethod
    def integrate(t, S, I, R, alpha, beta, gamma, injection_day, injection_ratio, K_inf, Nc):
        """Basic SIR model with multivariant capabilities"""

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        T = S + I + R

        # Compute infection pressure (IP) of both variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        IP_old = (1-alpha)*beta*np.matmul(Nc,(I/T))
        IP_new = alpha*K_inf*beta*np.matmul(Nc,(I/T))

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dS = - (IP_old + IP_new)*S
        dI = (IP_old + IP_new)*S - gamma*I
        dR = gamma*I

        # Update fraction of new COVID-19 variant
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if np.all((IP_old == 0)) and np.all((IP_new == 0)):
            dalpha = np.zeros(len(Nc))
        else:
            dalpha = IP_new/(IP_old+IP_new) - alpha

        # On injection_day, inject injection_ratio new strain to alpha
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if (t >= injection_day) & (alpha.sum().sum()==0):
            dalpha += injection_ratio

        return dS, dI, dR, dalpha

class COVID19_SEIQRD_hybrid_vacc(ODE):
    """
    The docstring will go here
    """

    # ...state variables and parameters
    states = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec', 'ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot','Inf_in','Inf_out','NH_R_in','C_R_in','ICU_R_in']
    parameters = ['beta', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc']
    stratified_parameters = [['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    dimensions = ['age_groups','doses']

    @staticmethod
    def integrate(t, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, Inf_in, Inf_out, NH_R_in, C_R_in, ICU_R_in,
                  beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc,
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):
        """
        Biomath extended SEIRD model for COVID-19
        *Deterministic implementation*
        """

        ###################
        ## Format inputs ##
        ###################

        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = (np.sum(f_VOC*K_hosp)*(h/(1-h)))/(1+ np.sum(f_VOC*K_hosp)*(h/(1-h)))
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klm_m(e_i,f_VOC) # Reduces from (n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klm_m(e_s,f_VOC)
        e_h = jit_matmul_klm_m(e_h,f_VOC)
        # Seasonality
        beta *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,0] + R[:,0]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,0]/VE
        f_R = R[:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,0] = - (N_vacc[:,0] + N_vacc[:,2])*f_S 
        dR[:,0] = - (N_vacc[:,0]+ N_vacc[:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,1] =  N_vacc[:,0]*f_S # 0 --> 1 dose
        dR[:,1] =  N_vacc[:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,2] =  N_vacc[:,2]*f_S # 0 --> 2 doses
        dR[:,2] =  N_vacc[:,2]*f_R 

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,1] + R[:,1]
        # Compute fraction of VE in states S and R
        f_S = S[:,1]/VE
        f_R = R[:,1]/VE
        # Compute transitioning in partially vaccinated circuit
        dS[:,1] = dS[:,1] - N_vacc[:,1]*f_S
        dR[:,1] = dR[:,1] - N_vacc[:,1]*f_R
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] + N_vacc[:,1]*f_S
        dR[:,2] = dR[:,2] + N_vacc[:,1]*f_R

        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,2] + R[:,2]
        # Compute fraction of VE in states S and R
        f_S = S[:,2]/VE
        f_R = R[:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] - N_vacc[:,3]*f_S
        dR[:,2] = dR[:,2] - N_vacc[:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,3] = dS[:,3] + N_vacc[:,3]*f_S
        dR[:,3] = dR[:,3] + N_vacc[:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        ################################
        ## calculate total population ##
        ################################

        T = np.expand_dims(np.sum(S_post_vacc + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R_post_vacc, axis=1),axis=1) # sum over doses

        #################################
        ## Compute system of equations ##
        #################################

        # Compute infection pressure (IP) of all variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = np.expand_dims( np.sum( np.outer(beta*s*jit_matmul_2D_1D(Nc, np.sum(((I+A)/T)*e_i, axis=1)), f_VOC*K_inf), axis=1), axis=1)

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS  = dS - IP*e_s*S_post_vacc
        dE  = IP*e_s*S_post_vacc - (1/sigma)*E
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - (1/da)*A
        dM_R = (1-h_acc)*((1-a)/omega)*I - (1/dm)*M_R
        dM_H = h_acc*((1-a)/omega)*I - (1/dhospital)*M_H
        dC_R = (1/dhospital)*c*(1-m_C)*M_H - (1/dc_R)*C_R
        dC_D = (1/dhospital)*c*m_C*M_H - (1/dc_D)*C_D
        dICU_star_R = (1/dhospital)*(1-c)*(1-m_ICU)*M_H - (1/dICU_R)*ICU_R
        dICU_star_D = (1/dhospital)*(1-c)*m_ICU*M_H - (1/dICU_D)*ICU_D
        dC_icurec = (1/dICU_R)*ICU_R - (1/dICUrec)*C_icurec
        dR  = dR + (1/da)*A + (1/dm)*M_R  + (1/dc_R)*C_R + (1/dICUrec)*C_icurec
        dD  = (1/dICU_D)*ICU_D + (1/dc_D)*C_D

        dM_in = ((1-a)/omega)*I - M_in
        dH_in = (1/dhospital)*M_H - H_in
        dH_tot = (1/dhospital)*M_H - (1/dc_R)*C_R - (1/dc_D)*C_D - (1/dICUrec)*C_icurec - (1/dICU_D)*ICU_D
        
        dNH_R_in = (1-h_acc)*((1-a)/omega)*I - NH_R_in
        dC_R_in = (1/dhospital)*c*(1-m_C)*M_H - C_R_in
        dICU_R_in = (1/dhospital)*(1-c)*(1-m_ICU)*M_H - ICU_R_in

        dInf_in = IP*e_s*S_post_vacc - Inf_in
        dInf_out = (1/da)*A + (1/dm)*M_R + (1/dc_R)*C_R + (1/dc_D)*C_D + (1/dICUrec)*C_icurec + (1/dICU_D)*ICU_D - Inf_out

        # Waning of natural immunity
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS[:,0] = dS[:,0] + zeta*R_post_vacc[:,0] 
        dR[:,0] = dR[:,0] - zeta*R_post_vacc[:,0]

        return (dS, dE, dI, dA, dM_R, dM_H, dC_R, dC_D, dC_icurec, dICU_star_R, dICU_star_D, dR, dD, dM_in, dH_in, dH_tot, dInf_in, dInf_out,dNH_R_in,dC_R_in,dICU_R_in)

class COVID19_SEIQRD_spatial_hybrid_vacc(ODE):

    # ...state variables and parameters
    states = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec','ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot']
    parameters = ['beta', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc', 'Nc_home', 'NIS']
    stratified_parameters = [['area', 'p'],['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    dimensions = ['NIS','age_groups','doses']

    @staticmethod
    def integrate(t, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, # time + SEIRD classes
                  beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc, Nc_home, NIS, # SEIRD parameters
                  area, p, # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D): 
        
        ###################
        ## Format inputs ##
        ###################
       
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = (np.sum(f_VOC*K_hosp)*(h/(1-h)))/(1+ np.sum(f_VOC*K_hosp)*(h/(1-h)))
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klmn_n(e_i,f_VOC) # Reduces from (n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klmn_n(e_s,f_VOC)
        e_h = jit_matmul_klmn_n(e_h,f_VOC)
        # Seasonality
        beta *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~
        # Compute vaccine eligible population
        VE = S[:,:,0] + R[:,:,0]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,0]/VE
        f_R = R[:,:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,:,0] = - (N_vacc[:,:,0] + N_vacc[:,:,2])*f_S 
        dR[:,:,0] = - (N_vacc[:,:,0]+ N_vacc[:,:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,:,1] =  N_vacc[:,:,0]*f_S # 0 --> 1 dose
        dR[:,:,1] =  N_vacc[:,:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,:,2] =  N_vacc[:,:,2]*f_S # 0 --> 2 doses
        dR[:,:,2] =  N_vacc[:,:,2]*f_R 

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,1] + R[:,:,1]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,1]/VE
        f_R = R[:,:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,:,1] = dS[:,:,1] - N_vacc[:,:,1]*f_S
        dR[:,:,1] = dR[:,:,1] - N_vacc[:,:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] + N_vacc[:,:,1]*f_S
        dR[:,:,2] = dR[:,:,2] + N_vacc[:,:,1]*f_R

        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,2] + R[:,:,2]
        # Compute fraction of VE in states S and R
        f_S = S[:,:,2]/VE
        f_R = R[:,:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,:,2] = dS[:,:,2] - N_vacc[:,:,3]*f_S
        dR[:,:,2] = dR[:,:,2] - N_vacc[:,:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,:,3] = dS[:,:,3] + N_vacc[:,:,3]*f_S
        dR[:,:,3] = dR[:,:,3] + N_vacc[:,:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        ################################
        ## calculate total population ##
        ################################

        T = np.sum(S_post_vacc + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R_post_vacc, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = S.shape[0] # spatial stratification

        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*NIS + np.identity(G)*(NIS @ (1-np.outer(p,p)))
        
        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.expand_dims(np.transpose(place_eff) @ T, axis=2)
        S_work = matmul_q_2D(np.transpose(place_eff), S_post_vacc)
        I_work = matmul_q_2D(np.transpose(place_eff), I)
        A_work = matmul_q_2D(np.transpose(place_eff), A)
        # The following line of code is the numpy equivalent of the above loop (verified)
        #S_work = np.transpose(np.matmul(np.transpose(S_post_vacc), place_eff))

        # Compute infectious work population (11,10)
        infpop_work = np.sum( (I_work + A_work)/T_work*e_i, axis=2)
        infpop_home = np.sum( (I + A)/np.expand_dims(T, axis=2)*e_i, axis=2)

        # Multiply with number of contacts
        multip_work = beta*np.expand_dims(jit_matmul_3D_2D(Nc_home, infpop_work), axis=2)
        multip_rest = beta*np.expand_dims(jit_matmul_3D_2D(Nc-Nc_home, infpop_home), axis=2)

        # Compute rates of change
        dS_inf = (S_work * multip_work + S_post_vacc * multip_rest)*e_s

        ############################
        ## Compute system of ODEs ##
        ############################

        dS  = dS - dS_inf
        dE  = dS_inf - (1/sigma)*E
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - (1/da)*A
        dM_R = (1-h_acc)*((1-a)/omega)*I - (1/dm)*M_R
        dM_H = h_acc*((1-a)/omega)*I - (1/dhospital)*M_H
        dC_R = (1/dhospital)*c*(1-m_C)*M_H - (1/dc_R)*C_R
        dC_D = (1/dhospital)*c*m_C*M_H - (1/dc_D)*C_D
        dICU_star_R = (1/dhospital)*(1-c)*(1-m_ICU)*M_H - (1/dICU_R)*ICU_R
        dICU_star_D = (1/dhospital)*(1-c)*m_ICU*M_H - (1/dICU_D)*ICU_D
        dC_icurec = (1/dICU_R)*ICU_R - (1/dICUrec)*C_icurec
        dR  = dR + (1/da)*A + (1/dm)*M_R  + (1/dc_R)*C_R + (1/dICUrec)*C_icurec
        dD  = (1/dICU_D)*ICU_D + (1/dc_D)*C_D

        dM_in = ((1-a)/omega)*I - M_in
        dH_in = (1/dhospital)*M_H - H_in
        dH_tot = (1/dhospital)*M_H - (1/dc_R)*C_R - (1/dc_D)*C_D - (1/dICUrec)*C_icurec - (1/dICU_D)*ICU_D

        ########################
        ## Waning of immunity ##
        ########################

        # Waning of natural immunity
        dS[:,:,0] = dS[:,:,0] + zeta*R_post_vacc[:,:,0] 
        dR[:,:,0] = dR[:,:,0] - zeta*R_post_vacc[:,:,0]      

        return (dS, dE, dI, dA, dM_R, dM_H, dC_R, dC_D, dC_icurec, dICU_star_R, dICU_star_D, dR, dD, dM_in, dH_in, dH_tot)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
