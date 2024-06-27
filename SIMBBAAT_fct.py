# Import required libraries
import numpy as np

# Functions to compute SIMBBAAT parameters
def Fct_m(L): 
    return L/2

def Fct_hr(er): 
    return er/2

def Fct_X(L,npts):
    m = Fct_m(L)
    return np.linspace(-m,m,npts)

def Fct_phi(f,w): 
    return f/w

def Fct_T_moy(f,L,w):
    phi = Fct_phi(f,w)
    return phi/L

def Fct_Gc(Ec,nuc):
    return Ec/(2*(1+nuc))

def Fct_eps(er,ec): 
    return ec/er

def Fct_alpha(L,er): 
    m = Fct_m(L)
    return m/er

def Fct_beta(er,ec,Er,Ec,nuc): 
    Gc = Fct_Gc(Ec,nuc)
    return np.sqrt(8*Gc*er/(ec*Er))

def Fct_beta_star(er,ec,Er,Ec,nur,nuc): 
    beta = Fct_beta(er,ec,Er,Ec,nuc)
    return np.sqrt(1-nur**2)*beta

def Fct_mu(L,er,ec,Er,Ec,nuc): 
    alpha = Fct_alpha(L,er)
    beta = Fct_beta(er,ec,Er,Ec,nuc)
    return alpha*beta

def Fct_mu_star(L,er,ec,Er,Ec,nur,nuc): 
    alpha = Fct_alpha(L,er)
    beta_star = Fct_beta_star(er,ec,Er,Ec,nur,nuc)
    return alpha*beta_star

def Fct_gamma(er,ec,Er,Ec): 
    return (6*Ec*er/(ec*Er))**(1/4)

def Fct_gamma_star(er,ec,Er,Ec,nur): 
    gamma = Fct_gamma(er,ec,Er,Ec)
    return (1-nur**2)**(1/4)*gamma

def Fct_lambda(L,er,ec,Er,Ec): 
    alpha = Fct_alpha(L,er)
    gamma = Fct_gamma(er,ec,Er,Ec)
    return alpha*gamma

def Fct_lambda_star(L,er,ec,Er,Ec,nur): 
    alpha = Fct_alpha(L,er)
    gamma_star = Fct_gamma_star(er,ec,Er,Ec,nur)
    return alpha*gamma_star

def Fct_Er_star(Er,nur): 
    return Er/(1-nur**2)

def Fct_Dr(w,er,Er): #?! pas de w
    return er**3*Er/12

def Fct_Dr_star(er,Er,nur): 
    Er_star = Fct_Er_star(Er,nur)
    return er**3*Er_star/12

def Fct_zetar(f,w,er,Er,nur): 
    phi = Fct_phi(f,w)
    Dr = Fct_Dr(w,er,Er)
    return np.sqrt(phi/Dr)

def Fct_zetar_star(f,w,er,Er,nur): 
    phi = Fct_phi(f,w)
    Dr_star = Fct_Dr_star(er,Er,nur)
    return np.sqrt(phi/Dr_star)

def Fct_zetar_star_hat(f,w,er,Er,nur,kb): 
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    return np.sqrt(1/kb)*zetar_star

#############################
# Membrane stress functions #
#############################

# Volkersen ###################################################################
def Fct_T_Volkersen(f,L,w,e1,e2,ec,E1,E2,Ec,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    #omega = eta*w
    phi = f/w
    T =  (1/(1+ksi))*(-c1*np.exp(-eta*x)+c2*np.exp(eta*x))*eta*phi
    return T

def Fct_N_Volkersen(f,L,e1,e2,ec,E1,E2,Ec,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    N2 = (1/(1+ksi))*(c1*np.exp(-eta*x)+c2*np.exp(eta*x)+ksi)*f
    N1 = f-N2
    return N1, N2

# Hart-Smith ##################################################################
def Fct_T_HartSmith(f,L,w,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,alphaT1,alphaT2,DeltaT,npts):
    x = np.linspace(0,L,npts)
    X = x-L/2
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    Gc = Gc/(1+3*kappa/2)
    ksi = e2*E2/(e1*E1)
    ksi_alpha = alphaT2/alphaT1
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    omega = eta*w
    phi = f/w
    T =  (1/(1+ksi))*(-c1*np.exp(-eta*x)+c2*np.exp(eta*x))*eta*phi + (1-1/ksi_alpha)/(1+ksi)*(np.sinh(eta*X)/np.cosh(omega))*eta*e2*E2*alphaT2*DeltaT
    return T

def Fct_N_HartSmith(f,L,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    Gc = Gc/(1+3*kappa/2)
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    N2 = (1/(1+ksi))*(c1*np.exp(-eta*x)+c2*np.exp(eta*x)+ksi)*f
    N1 = f-N2
    return N1, N2

# Demakles ####################################################################
def Fct_T_Demakles(f,L,w,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    Gc = Gc/(1+3*kappa/2)
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    #omega = eta*w
    phi = f/w
    T =  (1/(1+ksi))*(-c1*np.exp(-eta*x)+c2*np.exp(eta*x))*eta*phi
    return T

def Fct_N_Demakles(f,L,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    Gc = Gc/(1+3*kappa/2)
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    N2 = (1/(1+ksi))*(c1*np.exp(-eta*x)+c2*np.exp(eta*x)+ksi)*f
    N1 = f-N2
    return N1, N2

# Tsai ########################################################################
def Fct_T_Tsai(f,L,w,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    Gc = Gc/(1+kappa)
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    #omega = eta*w
    phi = f/w
    T =  (1/(1+ksi))*(-c1*np.exp(-eta*x)+c2*np.exp(eta*x))*eta*phi
    return T

def Fct_N_Tsai(f,L,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    Gc = Gc/(1+kappa)
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    N2 = (1/(1+ksi))*(c1*np.exp(-eta*x)+c2*np.exp(eta*x)+ksi)*f
    N1 = f-N2
    return N1, N2

# AdamsPeppiatt ###############################################################
def Fct_T_AdamsPeppiatt(f,L,w,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    eta = eta*np.sqrt(1/(1+3*kappa/2))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    #omega = eta*w
    phi = f/w
    T =  (1/(1+ksi))*(-c1*np.exp(-eta*x)+c2*np.exp(eta*x))*eta*phi
    return T

def Fct_N_AdamsPeppiatt(f,L,e1,e2,ec,E1,E2,Ec,nu1,nu2,nuc,npts):
    x = np.linspace(0,L,npts)
    Gc = Ec/(2*(1+nuc))
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))
    ksi = e2*E2/(e1*E1)
    eta = np.sqrt(Gc/ec*(1/(e1*E1)+1/(e2*E2)))
    kappa = (Gc/ec*((e1/G1)+(e2/G2)))/3
    eta = eta*np.sqrt(1/(1+3*kappa/2))
    c1 = -(1+ksi*np.exp(eta*L))/(2*np.sinh(eta*L))
    c2 = (1+ksi*np.exp(-eta*L))/(2*np.sinh(eta*L))
    N2 = (1/(1+ksi))*(c1*np.exp(-eta*x)+c2*np.exp(eta*x)+ksi)*f
    N1 = f-N2
    return N1, N2


############################
# Bending stress functions #
############################

# Goland and Reissner #########################################################
def Fct_k_GR(f,L,w,lr,er,Er,nur):
    m = Fct_m(L)
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    return 1/(1+2*np.sqrt(2)*np.tanh(m*zetar_star/(2*np.sqrt(2)))/np.tanh(lr*zetar_star))

def Fct_k_prime(f,L,w,lr,er,Er,nur):
    m = Fct_m(L)
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    k_GR = Fct_k_GR(f,L,w,lr,er,Er,nur)
    return 0.5*m*zetar_star*k_GR

def Fct_T_GR(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    T_moy = Fct_T_moy(f,L,w)
    mu = Fct_mu(L,er,ec,Er,Ec,nuc)
    k_GR = Fct_k_GR(f,L,w,lr,er,Er,nur)
    K1_GR = 0.25*(1+3*k_GR)*mu/np.sinh(mu)
    K2_GR = 0.75*(1-k_GR)
    T =  (K1_GR*np.cosh(mu*X/m)+K2_GR)*T_moy
    return T

def Fct_S_GR(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    T_moy = Fct_T_moy(f,L,w)
    k_GR = Fct_k_GR(f,L,w,lr,er,Er,nur)
    gamma_star = Fct_gamma_star(er,ec,Er,Ec,nur)
    lambda_star = Fct_lambda_star(L,er,ec,Er,Ec,nur)
    k_prime = Fct_k_prime(f,L,w,lr,er,Er,nur)
    R1 = np.cosh(lambda_star)*np.sin(lambda_star) + np.sinh(lambda_star)*np.cos(lambda_star)
    R2 = np.sinh(lambda_star)*np.cos(lambda_star) - np.cosh(lambda_star)*np.sin(lambda_star)
    R3 = 0.5*(np.sinh(2*lambda_star)+np.sin(2*lambda_star))
    K3_GR = (2*k_prime*np.cosh(lambda_star)*np.cos(lambda_star)+R2*lambda_star*k_GR)*gamma_star/R3
    K4_GR = (2*k_prime*np.sinh(lambda_star)*np.sin(lambda_star)+R1*lambda_star*k_GR)*gamma_star/R3
    S = (K3_GR*np.cos(lambda_star*X/m)*np.cosh(lambda_star*X/m) + K4_GR*np.sin(lambda_star*X/m)*np.sinh(lambda_star*X/m))*T_moy
    return S


# Hart-Smith ##################################################################
def Fct_rho_HS(er,ec,nur,kb):
    eps = Fct_eps(er,ec)
    return np.sqrt(0.25*(1+(3*(1-nur**2)/kb)*(1+eps)))

def Fct_k_HS(f,L,w,er,Er,nur,kb):
    m = Fct_m(L)
    zetar_star_hat = Fct_zetar_star_hat(f,w,er,Er,nur,kb)
    return 1/(1+(zetar_star_hat*m)+(zetar_star_hat*m)**2/6)

def Fct_T_HS(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts,kb):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    mu = Fct_mu(L,er,ec,Er,Ec,nuc)
    T_moy = Fct_T_moy(f,L,w)
    rho_HS = Fct_rho_HS(er,ec,nur,kb)
    k_HS = Fct_k_HS(f,L,w,er,Er,nur,kb)
    eps = Fct_eps(er,ec)
    K1_HS = (1+(3*(1-nur**2)/kb)*(1+eps)*k_HS)/(1+(3*(1-nur**2)/kb)*(1+eps))*(rho_HS*mu/np.sinh(rho_HS*mu))
    K2_HS = 1-(1+(3*(1-nur**2)/kb)*(1+eps)*k_HS)/(1+(3*(1-nur**2)/kb)*(1+eps))    
    T =  (K1_HS*np.cosh(rho_HS*mu*X/m)+K2_HS)*T_moy
    return T

def Fct_S_HS(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts,kb):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    T_moy = Fct_T_moy(f,L,w)
    alpha = Fct_alpha(L,er)
    gamma_star = Fct_gamma_star(er,ec,Er,Ec,nur)
    lambda_star = Fct_lambda_star(L,er,ec,Er,Ec,nur)
    eps = Fct_eps(er,ec)
    k_HS = Fct_k_HS(f,L,w,er,Er,nur,kb)
    K3_HS = 2*alpha*gamma_star**2*((np.cos(lambda_star)-np.sin(lambda_star))/np.exp(lambda_star))*(1+eps)*k_HS
    K4_HS = -((np.sin(lambda_star)+np.cos(lambda_star))/(np.sin(lambda_star)-np.cos(lambda_star)))*K3_HS
    S = (K3_HS*np.cos(lambda_star*X/m)*np.cosh(lambda_star*X/m) + K4_HS*np.sin(lambda_star*X/m)*np.sinh(lambda_star*X/m))*T_moy
    return S


# Ojalvo and Eidinoff #########################################################
def Fct_rho_OE(er,ec):
    eps = Fct_eps(er,ec)
    return np.sqrt((1+3*(1+eps)**2)/4)
    
def Fct_T_OE(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts,kb,k):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    T_moy = Fct_T_moy(f,L,w)
    eps = Fct_eps(er,ec)
    mu_star = Fct_mu_star(L,er,ec,Er,Ec,nur,nuc)
    rho_OE = Fct_rho_OE(er,ec)
    if k=="GR":
        k_M = Fct_k_GR(f,L,w,lr,er,Er,nur)
    elif k=="HS":
        k_M = Fct_k_HS(f,L,w,er,Er,nur,kb)
    else:
        k_M = Fct_k_GR(f,L,w,lr,er,Er,nur)
    K1_OE = (1+3*(1+eps)**2*k_M)/(1+3*(1+eps)**2)*(rho_OE*mu_star)/np.sinh(rho_OE*mu_star)
    K2_OE = 1-(1+3*(1+eps)**2*k_M)/(1+3*(1+eps)**2)
    T =  (K1_OE*np.cosh(rho_OE*mu_star*X/m)+K2_OE)*T_moy
    return T

def Fct_S_OE(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts,kb,k):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    T_moy = Fct_T_moy(f,L,w)
    eps = Fct_eps(er,ec)
    mu_star = Fct_mu_star(L,er,ec,Er,Ec,nur,nuc)
    lambda_star = Fct_lambda_star(L,er,ec,Er,Ec,nur)
    if k=="GR":
        k_M = Fct_k_GR(f,L,w,lr,er,Er,nur)
    elif k=="HS":
        k_M = Fct_k_HS(f,L,w,er,Er,nur,kb)
    else:
        k_M = Fct_k_GR(f,L,w,lr,er,Er,nur)
    n1 = np.sqrt(1-(3*eps*mu_star**2)/(16*lambda_star**2))
    n2 = np.sqrt(1+(3*eps*mu_star**2)/(16*lambda_star**2))
    lambda1_star = n1*lambda_star/m
    lambda2_star = n2*lambda_star/m
    a11 = (lambda2_star**2-lambda1_star**2)*np.cos(lambda1_star*m)*np.cosh(lambda2_star*m)-2*lambda1_star*lambda2_star*np.sin(lambda1_star*m)*np.sinh(lambda2_star*m)
    a12 = (lambda2_star**2-lambda1_star**2)*np.sin(lambda1_star*m)*np.sinh(lambda2_star*m)+2*lambda1_star*lambda2_star*np.cos(lambda1_star*m)*np.cosh(lambda2_star*m)
    a21 = -lambda1_star*(3*lambda2_star**2-lambda1_star**2-0.75*eps*(mu_star/m)**2)*np.sin(lambda1_star*m)*np.cosh(lambda2_star*m) \
          +lambda2_star*(lambda2_star**2-3*lambda1_star**2-0.75*eps*(mu_star/m)**2)*np.cos(lambda1_star*m)*np.sinh(lambda2_star*m)
    a22 = lambda2_star*(lambda2_star**2-3*lambda1_star**2-0.75*eps*(mu_star/m)**2)*np.sin(lambda1_star*m)*np.cosh(lambda2_star*m) \
         +lambda1_star*(3*lambda2_star**2-lambda1_star**2-0.75*eps*(mu_star/m)**2)*np.cos(lambda1_star*m)*np.sinh(lambda2_star*m)
    K3_OE = 2*((1-k_M)*a12+m*k_M*a22)/(a11*a22-a12*a21)*(1+eps)*er*(lambda_star/m)**4
    K4_OE = -2*((1-k_M)*a11+m*k_M*a21)/(a11*a22-a12*a21)*(1+eps)*er*(lambda_star/m)**4
    S = (K3_OE*np.cos(n1*lambda_star*X/m)*np.cosh(n2*lambda_star*X/m) + K4_OE*np.sin(n1*lambda_star*X/m)*np.sinh(n2*lambda_star*X/m))*T_moy
    return S


# Oplinger ####################################################################
def Fct_k_O(f,L,w,er,ec,Er,Ec,nur,nuc):
    eps = Fct_eps(er,ec)
    alpha = Fct_alpha(L,er)
    beta_star = Fct_beta_star(er,ec,Er,Ec,nur,nuc)
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    R = 2*np.sqrt(2)*er*zetar_star/beta_star
    R1_tilde = 2*np.sqrt(2)*np.sqrt(4*(1+3*eps/4)+R**2/4-np.sqrt((4*(1+3*eps/4)+R**2/4)**2-R**2))/R
    R2_tilde = np.sqrt(4*(1+3*eps/4)+R**2/4+np.sqrt((4*(1+3*eps/4)+R**2/4)**2-R**2))/(2*np.sqrt(2))
    m1 = er*zetar_star*R1_tilde/(2*np.sqrt(2))
    m2 = beta_star*R2_tilde
    C1 = ((2*R2_tilde)**2-1)/(64*(R2_tilde**2-(R/4)**2)*R2_tilde**2)
    C2 = ((2*R2_tilde)**2-1)/(48*R2_tilde**2)
    N_k_O = R1_tilde*(1+R**2*C2)+8*R2_tilde*R*(C1-C2)*np.tanh(m1*alpha)/np.tanh(m2*alpha)
    D_k_O = R1_tilde+8*R2_tilde*R*C1*np.tanh(m1*alpha)/np.tanh(m2*alpha)+2*np.sqrt(2)*(1+R**2*C1)*np.tanh(m1*alpha)
    return N_k_O/D_k_O

def Fct_T_O(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    Er_star = Fct_Er_star(Er,nur)
    alpha = Fct_alpha(L,er)
    beta_star = Fct_beta_star(er,ec,Er,Ec,nur,nuc)
    eps = Fct_eps(er,ec)
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    R = 2*np.sqrt(2)*er*zetar_star/beta_star
    R1_tilde = 2*np.sqrt(2)*np.sqrt(4*(1+3*eps/4)+R**2/4-np.sqrt((4*(1+3*eps/4)+R**2/4)**2-R**2))/R
    R2_tilde = np.sqrt(4*(1+3*eps/4)+R**2/4+np.sqrt((4*(1+3*eps/4)+R**2/4)**2-R**2))/(2*np.sqrt(2))
    m1 = er*zetar_star*R1_tilde/(2*np.sqrt(2))
    m2 = beta_star*R2_tilde
    C1 = ((2*R2_tilde)**2-1)/(64*(R2_tilde**2-(R/4)**2)*R2_tilde**2)
    C2 = ((2*R2_tilde)**2-1)/(48*R2_tilde**2)
    k_O = Fct_k_O(f,L,w,er,ec,Er,Ec,nur,nuc)
    N_K1_O = -(m1*beta_star)**2*Er_star*(1+R**2*C2-k_O)*m1
    D_K1_O = 16*(m1**2-beta_star**2/4)*(1+R**2*C1)
    K1_O = N_K1_O/D_K1_O
    N_K2_O = -(m2*beta_star)**2*Er_star*(C1*(1+R**2*C2-k_O)/(1+R**2*C1)-C2)*R**2*m2
    D_K2_O = 16*(m2**2-beta_star**2/4)
    K2_O = N_K2_O/D_K2_O
    T_O = K1_O*np.cosh(m1*alpha*X/m)/np.sinh(m1*alpha) + K2_O*np.cosh(m2*alpha*X/m)/np.sinh(m2*alpha)
    return T_O

def Fct_S_O(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    #phi = Fct_phi(f,w)
    T_moy = Fct_T_moy(f,L,w)
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    gamma_star = Fct_gamma_star(er,ec,Er,Ec,nur)
    lambda_star = Fct_lambda_star(L,er,ec,Er,Ec,nur)
    
    k_O = Fct_k_O(f,L,w,er,ec,Er,Ec,nur,nuc)
    k_prime = 0.5*m*zetar_star*k_O
    
    R1 = np.cosh(lambda_star)*np.sin(lambda_star) + np.sinh(lambda_star)*np.cos(lambda_star)
    R2 = np.sinh(lambda_star)*np.cos(lambda_star) - np.cosh(lambda_star)*np.sin(lambda_star)
    R3 = 0.5*(np.sinh(2*lambda_star)+np.sin(2*lambda_star))
    
    K3_GR = (2*k_prime*np.cosh(lambda_star)*np.cos(lambda_star)+R2*lambda_star*k_O)*gamma_star/R3
    K4_GR = (2*k_prime*np.sinh(lambda_star)*np.sin(lambda_star)+R1*lambda_star*k_O)*gamma_star/R3
    
    S_O = (K3_GR*np.cos(lambda_star*X/m)*np.cosh(lambda_star*X/m) + K4_GR*np.sin(lambda_star*X/m)*np.sinh(lambda_star*X/m))*T_moy
    return S_O


# Luo and Tong ################################################################
def Fct_k_LT(f,L,w,lr,er,ec,Er,Ec,nur,nuc):
    m = Fct_m(L)
    phi = Fct_phi(f,w)
    eps = Fct_eps(er,ec)
    Dr = Fct_Dr(w,er,Er)
    mu = Fct_mu(L,er,ec,Er,Ec,nuc)
    zetar = Fct_zetar(f,w,er,Er,nur)
    lmbda = Fct_lambda(L,er,ec,Er,Ec)
    beta_a1 = np.sqrt(0.5*((mu/m)**2+0.5*zetar**2+np.sqrt((mu/m)**4+0.5*(mu/m)**2*zetar**2+0.25*zetar**4)))
    beta_a2 = np.sqrt(0.5*((mu/m)**2+0.5*zetar**2-np.sqrt((mu/m)**4+0.5*(mu/m)**2*zetar**2+0.25*zetar**4)))
    beta_s1 = np.sqrt((lmbda/m)**2+zetar**2/8)
    beta_s2 = np.sqrt((lmbda/m)**2-zetar**2/8)
    b_11 = Dr*((beta_s1**2-beta_s2**2)*np.sinh(beta_s1*m)*np.sin(beta_s2*m)+2*beta_s1*beta_s2*np.cosh(beta_s1*m)*np.cos(beta_s2*m))
    b_12 = Dr*((beta_s1**2-beta_s2**2)*np.cosh(beta_s1*m)*np.cos(beta_s2*m)-2*beta_s1*beta_s2*np.sinh(beta_s1*m)*np.sin(beta_s2*m))
    b_21 = -Dr*(beta_s1*(beta_s1**2-3*beta_s2**2)*np.cosh(beta_s1*m)*np.sin(beta_s2*m)-beta_s2*(beta_s2**2-3*beta_s1**2)*np.sinh(beta_s1*m)*np.cos(beta_s2*m))
    b_22 = -Dr*(beta_s1*(beta_s1**2-3*beta_s2**2)*np.sinh(beta_s1*m)*np.cos(beta_s2*m)+beta_s2*(beta_s2**2-3*beta_s1**2)*np.cosh(beta_s1*m)*np.sin(beta_s2*m))
    K_a10 = 0.5*er*(mu/m)**2/(4*beta_a1**2-(mu/m)**2)
    K_a20 = 0.5*er*(mu/m)**2/(4*beta_a2**2-(mu/m)**2)
    Delta_a1 = ((1-beta_a1*m/np.tanh(beta_a1*m))/beta_a1**2-(1-beta_a2*m/np.tanh(beta_a2*m))/beta_a2**2)/(K_a20-K_a10)
    Delta_a2 = (K_a20*(1-beta_a1*m/np.tanh(beta_a1*m))/beta_a1**2-K_a10*(1-beta_a2*m/np.tanh(beta_a2*m))/beta_a2**2)/(K_a20-K_a10)
    Delta_s1 = np.sinh(beta_s1*m)*np.sin(beta_s2*m)-beta_s1*m*np.cosh(beta_s1*m)*np.sin(beta_s2*m)-beta_s2*m*np.sinh(beta_s1*m)*np.cos(beta_s2*m)
    Delta_s2 = np.cosh(beta_s1*m)*np.cos(beta_s2*m)-beta_s1*m*np.sinh(beta_s1*m)*np.cos(beta_s2*m)+beta_s2*m*np.cosh(beta_s1*m)*np.sin(beta_s2*m)
    Delta_prime = 2*(b_11*b_22-b_21*b_12)
    delta_phi = Delta_a1*phi/(er*(1+eps)*Er*er)
    delta_M = (0.5*Delta_a2/Dr+(b_22-b_12*zetar/np.tanh(zetar*lr))*Delta_s1/Delta_prime-(b_21-b_11*zetar/np.tanh(zetar*lr))*Delta_s2/Delta_prime)*phi
    return (1-delta_phi)/(1+zetar*m/np.tanh(zetar*lr)-delta_M)

def Fct_T_LT(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    hr = Fct_hr(er)
    X = Fct_X(L,npts)
    phi = Fct_phi(f,w)
    Gc = Fct_Gc(Ec,nuc)
    eps = Fct_eps(er,ec)
    mu = Fct_mu(L,er,ec,Er,Ec,nuc)
    zetar = Fct_zetar(f,w,er,Er,nur)
    beta_a1 = np.sqrt(0.5*((mu/m)**2+0.5*zetar**2+np.sqrt((mu/m)**4+0.5*(mu/m)**2*zetar**2+0.25*zetar**4)))
    beta_a2 = np.sqrt(0.5*((mu/m)**2+0.5*zetar**2-np.sqrt((mu/m)**4+0.5*(mu/m)**2*zetar**2+0.25*zetar**4)))
    K_a10 = 0.5*er*(mu/m)**2/(4*beta_a1**2-(mu/m)**2)
    K_a20 = 0.5*er*(mu/m)**2/(4*beta_a2**2-(mu/m)**2)
    k_LT = Fct_k_LT(f,L,w,lr,er,ec,Er,Ec,nur,nuc)  
    B_a1 = (12*K_a20*0.5*er*(1+eps)*k_LT-er**2)/(2*Er*er**3*beta_a1**2*(K_a20-K_a10)*np.sinh(beta_a1*m))*phi
    B_a3 = (12*K_a10*0.5*er*(1+eps)*k_LT-er**2)/(2*Er*er**3*beta_a2**2*(K_a10-K_a20)*np.sinh(beta_a2*m))*phi
    T_LT = 2*(Gc/ec)*((K_a10+hr)*beta_a1*B_a1*np.cosh(beta_a1*X)+(K_a20+hr)*beta_a2*B_a3*np.cosh(beta_a2*X))
    return T_LT

def Fct_S_LT(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    X = Fct_X(L,npts)
    phi = Fct_phi(f,w)
    eps = Fct_eps(er,ec)
    Dr = Fct_Dr(w,er,Er)
    zetar = Fct_zetar(f,w,er,Er,nur)
    lmbda = Fct_lambda(L,er,ec,Er,Ec)
    beta_s1 = np.sqrt((lmbda/m)**2+zetar**2/8)
    beta_s2 = np.sqrt((lmbda/m)**2-zetar**2/8)
    b_11 = Dr*((beta_s1**2-beta_s2**2)*np.sinh(beta_s1*m)*np.sin(beta_s2*m)+2*beta_s1*beta_s2*np.cosh(beta_s1*m)*np.cos(beta_s2*m))
    b_12 = Dr*((beta_s1**2-beta_s2**2)*np.cosh(beta_s1*m)*np.cos(beta_s2*m)-2*beta_s1*beta_s2*np.sinh(beta_s1*m)*np.sin(beta_s2*m))
    b_21 = -Dr*(beta_s1*(beta_s1**2-3*beta_s2**2)*np.cosh(beta_s1*m)*np.sin(beta_s2*m)-beta_s2*(beta_s2**2-3*beta_s1**2)*np.sinh(beta_s1*m)*np.cos(beta_s2*m))
    b_22 = -Dr*(beta_s1*(beta_s1**2-3*beta_s2**2)*np.sinh(beta_s1*m)*np.cos(beta_s2*m)+beta_s2*(beta_s2**2-3*beta_s1**2)*np.cosh(beta_s1*m)*np.sin(beta_s2*m))
    Delta_prime = 2*(b_11*b_22-b_21*b_12)  
    k_LT = Fct_k_LT(f,L,w,lr,er,ec,Er,Ec,nur,nuc)  
    B_s1 = (b_22-b_12*zetar/np.tanh(zetar*lr))/Delta_prime*0.5*er*(1+eps)*k_LT*phi
    B_s4 = (-b_21+b_11*zetar/np.tanh(zetar*lr))/Delta_prime*0.5*er*(1+eps)*k_LT*phi
    S_LT = 2*(Ec/ec)*(B_s1*np.sin(beta_s2*X)*np.sinh(beta_s1*X)+B_s4*np.cos(beta_s2*X)*np.cosh(beta_s1*X))
    return S_LT


# Zaho ########################################################################
def Fct_T_Z(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    hr = Fct_hr(er)
    hc = Fct_hr(ec)
    X = Fct_X(L,npts)
    phi = Fct_phi(f,w)
    Gc = Fct_Gc(Ec,nuc)    
    Er_hat = Er/(1-nur**2) # ou Er/(1-nur**2) def planes
    nur_hat = nur # ou nur/(1-nur**2) def planes    
    epsilon1 = (4*er**3+22*er**2*ec+39*er*ec**2)/(420*Er_hat) + (4*er**3+22*er**2*ec+39*er*ec**2)/(420*Er_hat)
    epsilon2 = ec/Gc + (4*er+3*ec+9*ec**2/er-15*nur_hat*ec)/(15*Er_hat) + (4*er+3*ec+9*ec**2/er-15*nur_hat*ec)/(15*Er_hat) 
    epsilon3 = (1+3*(1+ec/er)**2)/(Er_hat*er) + (1+3*(1+ec/er)**2)/(Er_hat*er)
    lambda1 = np.sqrt(0.5*(epsilon2/epsilon1+np.sqrt((epsilon2/epsilon1)**2-4*(epsilon3/epsilon1))))
    lambda2 = np.sqrt(0.5*(epsilon2/epsilon1-np.sqrt((epsilon2/epsilon1)**2-4*(epsilon3/epsilon1))))
    Delta = (epsilon1*lambda1**3-epsilon2*lambda1)*np.sinh(lambda1*m)*(np.sinh(lambda2*m)/(lambda2*m)-np.cosh(lambda2*m))-(epsilon1*lambda2**3-epsilon2*lambda2)*np.sinh(lambda2*m)*(np.sinh(lambda1*m)/(lambda1*m)-np.cosh(lambda1*m))
    M0 = Fct_k_GR(f,L,lr,w,er,Er,nur)*(hr+hc)*phi
    M1_0 = M0
    M2_L = M0
    a_1Z = (0.5*phi+3*(1/er+ec/er**2)*M1_0)/(Er_hat*er) - (0.5*phi+3*(1/er+ec/er**2)*M2_L)/(Er_hat*er) #?! signe
    a_2Z = -(0.5*phi+3*(1/er+ec/er**2)*M1_0)/(Er_hat*er) - (0.5*phi+3*(1/er+ec/er**2)*M2_L)/(Er_hat*er) #?! signe
    K_1Z = (a_2Z*(np.sinh(lambda2*m)/(lambda2*m)-np.cosh(lambda2*m))-0.5*(epsilon1*lambda2**3-epsilon2*lambda2)*np.sinh(lambda2*m)*phi/m)/Delta
    K_2Z = a_1Z/((epsilon1*lambda1**3-epsilon2*lambda1)*np.cosh(lambda1*m)-(epsilon1*lambda2**3-epsilon2*lambda2)*np.sinh(lambda1*m)/np.tanh(lambda2*m))
    K_3Z = (0.5*(epsilon1*lambda1**3-epsilon2*lambda1)*np.sinh(lambda1*m)*phi/m-a_2Z*(np.sinh(lambda1*m)/(lambda1*m)-np.cosh(lambda1*m)))/Delta
    K_4Z = a_1Z/((epsilon1*lambda2**3-epsilon2*lambda2)*np.cosh(lambda2*m)-(epsilon1*lambda1**3-epsilon2*lambda1)*np.sinh(lambda2*m)/np.tanh(lambda1*m))
    K_5Z = -(K_1Z*np.cosh(lambda1*m)+K_3Z*np.cosh(lambda2*m))
    T_Z = K_1Z*np.cosh(lambda1*X)+K_2Z*np.sinh(lambda1*X)+K_3Z*np.cosh(lambda2*X)+K_4Z*np.sinh(lambda2*X)+K_5Z
    return T_Z

def Fct_S_Z(f,L,w,lr,er,ec,Er,Ec,nur,nuc,npts):
    m = Fct_m(L)
    hr = Fct_hr(er)
    hc = Fct_hr(ec)
    X = Fct_X(L,npts)
    phi = Fct_phi(f,w)
    zetar_star = Fct_zetar_star(f,w,er,Er,nur)
    Er_hat = Er/(1-nur**2) # ou Er/(1-nur**2) def planes
    Ec_hat = Ec/(1-nur**2) # ou Ec/(1-nuc**2) def planes
    M0 = Fct_k_GR(f,L,lr,w,er,Er,nur)*(hr+hc)*phi
    V0 = zetar_star*Fct_k_GR(f,L,lr,w,er,Er,nur)*m/2;    
    M1_0 = M0
    M2_L = M0
    V1_0 = -V0
    V2_L = -V0
    epsilon3 = (1+3*(1+ec/er)**2)/(Er_hat*er) + (1+3*(1+ec/er)**2)/(Er_hat*er)
    epsilon4 = ec/Ec_hat + 13*(er/Er_hat+er/Er_hat)/55 #?! 13/35 (TI et Eric) ou 13/55 (Seb)
    epsilon5 = 12*(1/(Er_hat*er)+1/(Er_hat*er))/5
    epsilon6 = 12*(1/(Er_hat*er**3)+1/(Er_hat*er**3))    
    lambda3 = np.sqrt(0.25*epsilon5/epsilon4+np.sqrt(0.25*epsilon6/epsilon4))
    lambda4 = np.sqrt(-0.25*epsilon5/epsilon4+np.sqrt(0.25*epsilon6/epsilon4))    
    b_1Z = 6*M2_L/(Er_hat*er**3) + 6*M1_0/(Er_hat*er**3)
    b_2Z = 6*M2_L/(Er_hat*er**3) - 6*M1_0/(Er_hat*er**3)
    b_3Z = 6*V1_0/(Er_hat*er**3) - 6*V2_L/(Er_hat*er**3)
    b_4Z = -6*V1_0/(Er_hat*er**3) - 6*V2_L/(Er_hat*er**3)
    c1 = epsilon4*(lambda3**2-lambda4**2) - epsilon5
    c2 = 2*epsilon4*lambda3*lambda4
    c3 = lambda3*(epsilon3*(lambda3**2-3*lambda4**2) - epsilon5) #?! lambda3*(epsilon4*(lambda3**2-3*lambda4**2) - epsilon5) (TI et Eric)
    c4 = lambda4*(epsilon4*(lambda4**2-3*lambda3**2) + epsilon5)
    R_21 = c1*np.cosh(lambda3*m)*np.cos(lambda4*m) - c2*np.sinh(lambda3*m)*np.sin(lambda4*m)
    R_22 = c1*np.sinh(lambda3*m)*np.sin(lambda4*m) + c2*np.cosh(lambda3*m)*np.cos(lambda4*m)
    R_23 = c3*np.sinh(lambda3*m)*np.cos(lambda4*m) + c4*np.cosh(lambda3*m)*np.sin(lambda4*m)
    R_24 = c3*np.cosh(lambda3*m)*np.sin(lambda4*m) - c4*np.sinh(lambda3*m)*np.cos(lambda4*m)
    R_25 = c1*np.cosh(lambda3*m)*np.sin(lambda4*m) + c2*np.sinh(lambda3*m)*np.cos(lambda4*m)
    R_26 = c1*np.sinh(lambda3*m)*np.cos(lambda4*m) - c2*np.cosh(lambda3*m)*np.sin(lambda4*m)
    R_27 = c3*np.sinh(lambda3*m)*np.sin(lambda4*m) - c4*np.cosh(lambda3*m)*np.cos(lambda4*m)
    R_28 = c3*np.cosh(lambda3*m)*np.cos(lambda4*m) + c4*np.sinh(lambda3*m)*np.sin(lambda4*m)
    K_6Z = (b_1Z*R_24-b_4Z*R_22)/(R_21*R_24-R_22*R_23)
    K_7Z = (b_3Z*R_25-b_2Z*R_27)/(R_25*R_28-R_26*R_27)
    K_8Z = (b_2Z*R_28-b_3Z*R_26)/(R_25*R_28-R_26*R_27)
    K_9Z = (b_4Z*R_21-b_1Z*R_23)/(R_21*R_24-R_22*R_23)
    S_Z = K_6Z*np.cos(lambda4*X)*np.cosh(lambda3*X)+K_7Z*np.cos(lambda4*X)*np.sinh(lambda3*X)+K_8Z*np.sin(lambda4*X)*np.cosh(lambda3*X)+K_9Z*np.sin(lambda4*X)*np.sinh(lambda3*X)
    return S_Z