import numpy as np
import pylab as py
from scipy.integrate import quad
py.ion()


T_0 = 2.725
eV2K = 11604.5
norm_factor = 3.*(7/8.)*(4/11.)**(4/3.) * 4.64e-34/(1.783e-33)
h = 0.6731
rho_crit = 1.879e-29*h**2./1.783e-33


def integrand(x,a,m):
  return x**2 * np.sqrt(x**2 * (4./11.)**(2/3.) * T_0**2 * a**(-2) + (m*eV2K)**2.)/(np.exp(x) + 1.)

def my_integral(a,m):
  return quad(integrand, 0, 100., args=(a,m))[0]



################

a_min = 1e-6
a_max = 1.0
crit_normalize = True



#Calculate logarithmic interval for a.
dlog10_a = (0 - np.log10(a_min))/1000.
m = 0.0

rho_nu_0 = []
rho_nu_0p05 = []
rho_nu_0p5 = []
scale_factor = []

for i in range(1001):
    a = 10**(-6 + dlog10_a*i)
    scale_factor.append(a)

    rho_nu_0.append( my_integral(a,0.0/3.) * T_0**3. * (4/11./np.pi**2.)/a**3)
    rho_nu_0p05.append( my_integral(a,0.05/3.) * T_0**3. * (4/11./np.pi**2.)/a**3)
    rho_nu_0p5.append( my_integral(a,0.5/3.) * T_0**3. * (4/11./np.pi**2.)/a**3)
    
scale_factor = np.array(scale_factor)
z = 1/scale_factor - 1.

rho_nu_0 = np.array(rho_nu_0)
rho_nu_0 *= norm_factor/a_min**4./rho_nu_0[0]/rho_crit
rho_nu_0p05 = np.array(rho_nu_0p05)
rho_nu_0p05 *= norm_factor/a_min**4./rho_nu_0p05[0]/rho_crit
rho_nu_0p5 = np.array(rho_nu_0p5)
rho_nu_0p5 *= norm_factor/a_min**4./rho_nu_0p5[0]/rho_crit


#Calculate other components energy densities.
rho_gamma = 4.64e-34/1.783e-33 / scale_factor**4/rho_crit
rho_matter = (0.02222+0.1197)/h**2/scale_factor**3.
rho_lambda_0 = (1. - rho_gamma[-1] - rho_matter[-1] - rho_nu_0[-1])*np.ones(len(scale_factor))
rho_lambda_0p05 = (1. - rho_gamma[-1] - rho_matter[-1] - rho_nu_0p05[-1])*np.ones(len(scale_factor))
rho_lambda_0p5 = (1. - rho_gamma[-1] - rho_matter[-1] - rho_nu_0p5[-1])*np.ones(len(scale_factor))

rho_sum_0 = (rho_gamma + rho_matter + rho_lambda_0 + rho_nu_0)
rho_sum_0p05 = (rho_gamma + rho_matter + rho_lambda_0p05 + rho_nu_0p05)
rho_sum_0p5 = (rho_gamma + rho_matter + rho_lambda_0p5 + rho_nu_0p5)

#Normalize by rho_crit(a)
if crit_normalize:
  omega_gamma_0 = rho_gamma/rho_sum_0
  omega_matter_0 = rho_matter/rho_sum_0
  omega_lambda_0 = rho_lambda_0/rho_sum_0
  omega_nu_0 = rho_nu_0/rho_sum_0

  omega_gamma_0p05 = rho_gamma/rho_sum_0p05
  omega_matter_0p05 = rho_matter/rho_sum_0p05
  omega_lambda_0p05 = rho_lambda_0p05/rho_sum_0p05
  omega_nu_0p05 = rho_nu_0p05/rho_sum_0p05

  omega_gamma_0p5 = rho_gamma/rho_sum_0p5
  omega_matter_0p5 = rho_matter/rho_sum_0p5
  omega_lambda_0p5 = rho_lambda_0p5/rho_sum_0p5
  omega_nu_0p5 = rho_nu_0p5/rho_sum_0p5

py.rc('axes', linewidth=1)
py.rc('font', family='sans-serif')


f, (ax1) = py.subplots(1, 1, sharex=True, figsize=(7,8), dpi=100)
ax1.loglog(scale_factor, omega_gamma_0, 'r-', linewidth=2, label='Photons (CMB)')
ax1.loglog(scale_factor, omega_matter_0, 'b-', linewidth=2, label='CDM + Baryons')
ax1.loglog(scale_factor, omega_lambda_0, 'm-', linewidth=2, label='Dark Energy')
ax1.loglog(scale_factor, omega_nu_0, 'k-', linewidth=2, label='$\\Sigma m_\\nu=0$ eV')
ax1.loglog(scale_factor, omega_nu_0p05, 'k--', linewidth=2, label='$\\Sigma m_\\nu=0.05$ eV')
ax1.loglog(scale_factor, omega_nu_0p5, 'k-.', linewidth=2, label='$\\Sigma m_\\nu=0.5$ eV')
ax1.set_xlabel('Scale Factor $a$', fontsize=20)


#Add a second x-axis showing redshift instead of scale factor.
wanted_z = np.array([1e5, 1e4, 1e4, 1e3, 1e2, 1e1, 1, 0], dtype=np.float)
these_a = 1./(1.+wanted_z)
a_ticks = (np.log10(these_a) - np.log10(a_min))/6.
ax3 = ax1.twiny()
ax3.set_xticks(a_ticks)
ax3.set_xbound(ax1.get_xbound())
ax3.set_xticklabels(["%d" % x for x in wanted_z])
ax3.set_xlabel('Redshift $z$', fontsize=20)

ax1.set_ylabel('$\\rho_x/\\rho_{cr}$', fontsize=20)
ax1.set_ylim((1e-5,2))
ax1.legend(loc='best', labelspacing=1)
py.draw()
py.tight_layout()
py.subplots_adjust(hspace=0.0)
py.savefig('omegas_changing_neutrino_mass.pdf')




f, (ax2) = py.subplots(1, 1, sharex=True, figsize=(8,8), dpi=100)
ax2.loglog(scale_factor, rho_gamma*rho_crit, 'r-', linewidth=2, label='Photons (CMB)')
ax2.loglog(scale_factor, rho_matter*rho_crit, 'b-', linewidth=2, label='CDM + Baryons')
ax2.loglog(scale_factor, rho_lambda_0*rho_crit, 'm-', linewidth=2, label='Dark Energy')
ax2.loglog(scale_factor, rho_nu_0*rho_crit, 'k-', linewidth=2, label='$\\Sigma m_\\nu=0$ eV')
ax2.loglog(scale_factor, rho_nu_0p05*rho_crit, 'k--', linewidth=2, label='$\\Sigma m_\\nu=0.05$ eV')
ax2.loglog(scale_factor, rho_nu_0p5*rho_crit, 'k-.', linewidth=2, label='$\\Sigma m_\\nu=0.5$ eV')
ax2.set_xlabel('Scale Factor $a$', fontsize=20)

#Add a second x-axis showing redshift instead of scale factor.
wanted_z = np.array([1e5, 1e4, 1e4, 1e3, 1e2, 1e1, 1, 0], dtype=np.float)
these_a = 1./(1.+wanted_z)
a_ticks = (np.log10(these_a) - np.log10(a_min))/6.
ax4 = ax2.twiny()
ax4.set_xticks(a_ticks)
ax4.set_xbound(ax2.get_xbound())
ax4.set_xticklabels(["%d" % x for x in wanted_z])
ax4.set_xlabel('Redshift $z$', fontsize=20)
                    
ax2.set_ylabel('Energy Density [eV/cm$^3$]', fontsize=20)
ax2.legend(loc='best', labelspacing=1)
py.draw()
py.tight_layout()
py.subplots_adjust(hspace=0.0)
py.savefig('energy_density_changing_neutrino_mass.pdf')
