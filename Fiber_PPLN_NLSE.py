"""The absorption coefficient of PPLN (Alpha), gaussian beam calculation (gbeam_approx), chi2 parameter,
and the nonlinear operator for three wave mixing is taken from Alex's Matlab code """

import numpy as np
from pynlo.light.DerivedPulses import SechPulse
from pynlo.media.fibers.fiber import FiberInstance
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.special import erf
from pynlo.interactions.FourWaveMixing.SSFM import SSFM
from pynlo.media.crystals.XTAL_PPLN import DengSellmeier
from numpy.fft import fftshift, fft
from scipy.integrate import simps
from pynlo.media.fibers.calculators import DTabulationToBetas
import copy


# prevent divide by zero errors
def num_over_denom(num, denom):
    """
    :param num: numerator
    :param denom: denominator
    :return: 0 where the denominator is less than 1e-15, otherwise num / (denom + 1e-20)

    prevents divide by zero errors
    """
    return np.where(abs(denom) > 1e-15, num / (denom + 1e-20), 0.0)


def get_betas_from_Dcurve(lamda_nm, D_psnmkm, pulse, polyOrder=2):
    """
    :param lamda_nm: wavelength axis in nm
    :param D_psnmkm: D value at each wavelength
    :param pulse: pulse (used for center wavelength in series expansion)
    :return: [beta2, beta3, ...] up to polyorder
    """
    return DTabulationToBetas(pulse.center_wavelength_nm, np.vstack((lamda_nm, D_psnmkm)).T, polyOrder,
                              DDataIsFile=False, return_diagnostics=False)


# absorption coefficient of PPLN
# angular frequency in THz -> absorption coefficient (1/m)
def Alpha(W_THz):
    """
    :param W_THz: angular frequency in THz
    :return: absorption coefficient (1/m)

    I got this equation from Alex's code
    """
    return 1e6 * (1 + erf(-(W_THz - 300.) / (10 * np.sqrt(2))))


# return the current sign of chi2
# z pos (m), length of crystal (m), poling period (m) -> +- 1
# the poling period should be an array such as np.linspace(starting_period, ending_period, 5000)
def grating(z, L, poling_period_mks):
    """
    :param z: float
    :param L: total length of crystal
    :param poling_period_mks: 1D array poling period (m)
    :return: +- 1

    This function takes the poling period that is a 1D array and the length of the crystal L, to determine
    the sign of chi2 at a given z (float). It does this via interpolation
    """
    zD = np.linspace(0, L, len(poling_period_mks))

    period = interp1d(zD, poling_period_mks)(z)
    return np.sign(np.cos(2 * np.pi * z / period))


# calculates w/w0 where w is the beam radius
# assumes a gaussian beam with a focus at the center of the crystal
# pulse instance, length of crystal (m), z pos (m) -> float
def gbeam_approx(pulse, L, z, r_um=15):
    """
    :param pulse: pulse instance (for center wavelength)
    :param L: length of the crystal (m)
    :param z: position inside the crystal (m)
    :param r_um: radius at the beam focus
    :return: w/w0 (float)

    calculating wo/w (gaussian beam), assuming a focus at the center of the crystal, this is a normalization
    factor for the overall intensity

    the chi2 parameter is scaled by 1 / sqrt(Aeff), where Aeff is effective area. We calculate this at the start of the
    simulation and initialize chi 2 using Aeff at the focus. From there we scale chi2 by w/w0 at each position
    in the crystal (wo/w should be 1 t the center of the crystal).
    """
    center_wl = pulse.center_wavelength_nm * 1e-9

    w_0 = np.pi * (r_um * 1e-6) ** 2 / center_wl
    return 1 / np.sqrt(1 + ((z - L / 2) / w_0) ** 2)


# Pulse Class
# this is an altered version of pynlo's Pulse class
class Pulse(SechPulse):
    # setting default values to a sech pulse with T0 = 200 fs
    # center_wavelength = 1550 nm, no chirp
    # a time window of 10ps and 2**14 points in the simulation,
    # a repetition rate of 100 MHz, and pulse energy of 5 nJ (0.5 W average power)
    def __init__(self, T0_ps=.2,
                 center_wavelength_nm=1550,
                 time_window_ps=10,
                 GDD=0,
                 TOD=0,
                 NPTS=2 ** 14,
                 frep_MHz=100,
                 EPP_nJ=5.0):
        """
        :param T0_ps: pulse duration (for creating sech pulse)
        :param center_wavelength_nm:
        :param time_window_ps:
        :param GDD:
        :param TOD:
        :param NPTS:
        :param frep_MHz:
        :param EPP_nJ:
        """

        super().__init__(power=1,  # Power will be scaled by set_epp
                         T0_ps=T0_ps,
                         center_wavelength_nm=center_wavelength_nm,
                         time_window_ps=time_window_ps,
                         GDD=GDD, TOD=TOD,
                         NPTS=NPTS,
                         frep_MHz=frep_MHz,
                         power_is_avg=False)

        # set the pulse energy, and add an attribute of wavelength in micron (nice to have)
        self.wl_um = self.wl_nm * 1e-3
        self.set_epp(EPP_nJ * 1e-9)

        # create this attribute so that later if a new field is set, it can be scaled to what the original pulse
        # energy was.
        self.desired_epp = EPP_nJ * 1e-9

    # when setting the pulse energy, also have it update the desired_epp attribute
    def set_epp(self, desired_epp_J):
        """
        :param desired_epp_J:
        :return:
        """
        super().set_epp(desired_epp_J)
        self.desired_epp = desired_epp_J

    # set a new field based on: time (ps), electric field in time domain
    # utilizes the pulse class's already built in set_AT function
    def set_AT_experiment(self, T_ps, AT):
        """
        :param T_ps:
        :param AT:
        :return:
        """
        # It's important you interpolate amplitude and phase rather than real and imaginary, because those two
        # methods are different, and the latter results in oscillations in the amplitude!
        gridded_amplitude = interp1d(T_ps, abs(AT), kind='linear', bounds_error=False, fill_value=0)
        gridded_phase = interp1d(T_ps, np.unwrap(np.arctan2(AT.imag, AT.real)), kind='linear', bounds_error=False,
                                 fill_value=0)
        amplitude = gridded_amplitude(self.T_ps)
        phase = gridded_phase(self.T_ps)

        # the interpolated electric field on the pulse's time grid
        AT_new = amplitude * np.exp(1j * phase)

        # set the pulse's electric field to the new one
        self.set_AT(AT_new)

        # rescale the electric field to keep the pulse energy unchanged
        self.set_epp(self.desired_epp)

    # set a new field based on: lambda (um), electric field in frequency domain
    # utilizes the pulse class's already built in set_AW function
    def set_AW_experiment(self, wl_um, AW):
        """
        :param wl_um:
        :param AW:
        :return:
        """
        # It's important you interpolate amplitude and phase rather than real and imaginary, because those two
        # methods are different, and the latter results in oscillations in the amplitude!
        gridded_amplitude = interp1d(wl_um, abs(AW), kind='linear', bounds_error=False, fill_value=0)
        gridded_phase = interp1d(wl_um, np.unwrap(np.arctan2(AW.imag, AW.real)), kind='linear', bounds_error=False,
                                 fill_value=0)
        amplitude = gridded_amplitude(self.wl_um)
        phase = gridded_phase(self.wl_um)

        # the interpolated electric field on the pulse's time grid
        AW_new = amplitude * np.exp(1j * phase)

        # set the pulse's electric field to the new one
        self.set_AW(AW_new)

        # rescale the electric field to keep the pulse energy unchanged
        self.set_epp(self.desired_epp)


# PPLN medium class
# this is an altered version of pynlo's FiberInstance class
class PPLN(FiberInstance):
    def __init__(self):
        super().__init__()

    # set the refractive index to that of PPLN
    def set_refractive_index(self, lda_um, n):
        lda_nm = lda_um * 1e3

        # this entry in the dict specifies what self.x and self.y
        # it is called up later in the self.get_betas function
        self.fiberspecs["dispersion_format"] = "n"

        self.x, self.y = lda_nm, n

    # the equivalent of the FiberInstance's self.generate_fiber function
    def generate_ppln(self, pulse, length, center_wl_nm, poling_period_um, r_um=15, T_C=24.5):
        # calculate chi2_parameter based on Alex's Matlab code
        w0 = pulse.center_frequency_THz * 2 * np.pi
        deff = 19.6e-12
        chi2 = 2 * deff
        e0 = 8.85e-12
        Aeff = np.pi * (r_um * 1e-6) ** 2
        reference_lamda = np.linspace(.3, 6, 5000) * 1e3  # nm
        LNJundt = DengSellmeier(T_C).n  # using T = 24.5 C cause that's what Alex did
        n0 = interp1d(reference_lamda, LNJundt(reference_lamda))(pulse.center_wavelength_nm)
        chi2_param = (1 / 4) * (chi2 / n0) * (w0 * 1e12 / self.c_mks) * np.sqrt(2 / (e0 * self.c_mks * Aeff))

        # set the crystal length, center_wavelength, poling_period, gain, and refractive index
        self.length = length
        self.center_wavelength = center_wl_nm
        self.poling_period_mks = poling_period_um * 1e-6
        self.gain = - Alpha(pulse.W_THz)
        self.fiberspecs['is_gain'] = True
        self.fiberspecs['gain_x_data'] = None
        self.set_refractive_index(reference_lamda / 1000, LNJundt(reference_lamda))

        # the gamma (chi2_parameter) changes with z
        def gamma_function(z):
            return chi2_param * grating(z, self.length, self.poling_period_mks)

        self.set_gamma_function(gamma_function)

        self.r_um = r_um

    # alter the self.get_betas function so that it stores beta0 and beta1 which will be needed in the
    # three wave mixing nonlinear operator later.
    def get_betas(self, pulse, z=0):
        supplied_W_THz = 2 * np.pi * 1e-12 * 3e8 / (self.x * 1e-9)
        supplied_betas = self.y * 2 * np.pi / (self.x * 1e-9)

        # InterpolatedUnivariateSpline wants increasing x, so flip arrays
        interpolator = InterpolatedUnivariateSpline(supplied_W_THz[::-1], supplied_betas[::-1])
        B = interpolator(pulse.W_THz)
        center_index = np.argmin(np.abs(pulse.V_THz))
        slope = np.gradient(B) / np.gradient(pulse.W_THz)
        self.beta0 = B[center_index]
        self.beta1 = slope[center_index]
        return super().get_betas(pulse, z)


# Three Wave Mixing Class
# based on pynlo's Four Wave Mixing class
class PPLNThreeWaveMixing(SSFM):
    def __init__(self):
        super().__init__()

    # when loading the PPLN parameters, also record beta0, beta1, and the gamma_function from the PPLN instance
    # create a self.gbeam_approx method that calculates w/w0 for a given z position (w here is the beam radius)
    def load_fiber_parameters(self, pulse_in, fiber, output_power, z=0):
        super().load_fiber_parameters(pulse_in, fiber, output_power, z)
        self.beta0 = fiber.beta0
        self.beta1 = fiber.beta1
        self.gamma_function = fiber.gamma_function
        self.gbeam_approx = lambda z: gbeam_approx(pulse_in, fiber.length, z, fiber.r_um)

    # when setting up the simulation, also create self.T_ps (the pulse's time grid in ps) which will
    # be needed later for the nonlinear operator
    def setup_fftw(self, pulse_in, fiber, output_power, raman_plots=False):
        super().setup_fftw(pulse_in, fiber, output_power, raman_plots)
        self.T_ps = pulse_in.T_ps

    # this is an altered version of the Four Wave Mixing class's integrate_over_dz method
    # I changes two things: 1) update the current z-position (self.z), 2) instead of integrating over a
    # distance delta_z, integrate from a given z start to a given z finish, I think the ability to pass in
    # a zstart parameter makes the chi(z) more accurate.
    def integrate_over_dz(self, zstart, zfinish, direction=1):
        dist = zfinish - zstart
        dz = self.dz

        self.last_h = -1.0  # Force an update of exp_D
        force_last_dz = False
        factor = 2 ** (1.0 / self.eta)

        if (2.0 * dz > dist):
            dz = dist / 2.0

        # I changed the while condition, the original one was while dist>0:
        while self.z < zfinish:
            self.Ac[:] = self.A
            self.Af[:] = self.A

            self.Ac[:] = self.Advance(self.Ac, 2.0 * dz, direction)
            self.Af[:] = self.Advance(self.Af, dz, direction)
            self.Af[:] = self.Advance(self.Af, dz, direction)
            # delta = |Af - Ac| / |Af|
            delta = self.CalculateLocalError()

            old_dz = dz
            new_dz = dz
            if not self.suppress_iteration:
                print("iteration:", self.iter, "dz:", dz, "distance:", dist, "local error", delta)

            if delta > 2.0 * self.local_error:
                # Discard the solution, decrease step
                new_dz = dz / 2.0
                if new_dz >= self.dz_min:
                    dz = new_dz
                    # discard current step
                    continue
                else:
                    # accept step after all
                    pass
            elif (delta >= self.local_error) and (delta <= 2.0 * self.local_error):
                # Keep solution, decrease step
                new_dz = dz / factor
                if new_dz >= self.dz_min:
                    dz = new_dz
                else:
                    pass
            elif (delta >= (0.5 * self.local_error)) and (delta <= self.local_error):
                # keep the step
                new_dz = new_dz
            else:  # delta < local_error/2
                # Step too small
                new_dz = dz * factor
                dz = new_dz
            if self.eta == 3:
                self.A[:] = (4.0 / 3.0) * self.Af - (1.0 / 3.0) * self.Ac
            elif self.eta == 5:
                self.A[:] = (16.0 / 15.0) * self.Af - (1.0 / 15.0) * self.Ac
            else:
                p = 2.0 ** (self.eta - 1.0)
                self.A[:] = (p / (p - 1.0)) * self.Af - (1.0 / (p - 1.0)) * self.Ac

            dist -= 2.0 * old_dz
            self.iter += 1
            self.z += 2.0 * old_dz  # I added this

            if (2.0 * dz > dist) and (dist > 2.0 * self.dz_min):
                force_last_dz = True
                return_dz = dz
                dz = dist / 2.0

        if force_last_dz:
            dz = return_dz
        self.dz = dz

    # this is an altered version of the propagate function in the Four Wave Mixing Class
    # I changed two things: 1) pass self.integrate_over_dz a zstart and zfinish instead of delta_z,
    # 2) return a class instead of a tuple (nice to have)
    def propagate(self, pulse_in, fiber, n_steps, output_power=None, reload_fiber_each_step=False):
        n_steps = int(n_steps)

        # Copy parameters from pulse and fiber into class-wide variables
        # I changed the original z_positions which was an arange with n_steps+1 entry
        # I added zs which will record the actual z positions where the spectrum was returned
        z_positions = np.linspace(0, fiber.length, n_steps)
        zs = np.zeros(len(z_positions))
        if n_steps == 1:
            delta_z = fiber.length
        else:
            delta_z = z_positions[1] - z_positions[0]

        AW = np.complex64(np.zeros((pulse_in.NPTS, n_steps)))
        AT = np.complex64(np.copy(AW))
        AW[:, 0] = pulse_in.AW
        AT[:, 0] = pulse_in.AT

        print("Pulse energy before", fiber.fibertype, ":",
              1e9 * pulse_in.calc_epp(), 'nJ')

        # using deepcopy instead of the Pulse's clone function
        # I had some issue using the clone function that might have had to do with the number of projects
        # I had open in the same Pycharm window
        pulse_out = copy.deepcopy(pulse_in)
        self.setup_fftw(pulse_in, fiber, output_power)
        self.load_fiber_parameters(pulse_in, fiber, output_power)

        # create the self.z parameter and set it to zero
        self.z = 0
        for i in range(1, n_steps):
            # zstart is the current z position
            zstart = self.z
            # zfinish is the next position where I want to record the spectrum
            zfinish = z_positions[i]

            print("Step:", i, "Distance remaining:", fiber.length * (1 - np.float(i) / n_steps))

            if reload_fiber_each_step:
                self.load_fiber_parameters(pulse_in, fiber, output_power, z=i * delta_z)

            self.integrate_over_dz(zstart, zfinish)
            AW[:, i] = self.conditional_ifftshift(self.FFT_t_2(self.A))
            AT[:, i] = self.conditional_ifftshift(self.A)
            zs[i] = self.z  # I added this
            pulse_out.set_AT(self.conditional_ifftshift(self.A))
            print("Pulse energy after:",
                  1e9 * pulse_out.calc_epp(), 'nJ')

        pulse_out.set_AT(self.conditional_ifftshift(self.A))

        print("Pulse energy after", fiber.fibertype, ":",
              1e9 * pulse_out.calc_epp(), 'nJ')
        self.cleanup()

        # I added this, originaly it would have returned z_positions, AW, AT, pulse_out
        class res:
            def __init__(self):
                self.zs = zs
                self.AW = AW.T
                self.AT = AT.T
                self.pulse = pulse_out

        return res()

    # Nonlinear operator for Three Wave Mixing
    # over rides the original Four Wave Mixing Nonlinear Operator
    def NonlinearOperator(self, A):
        self.Aw[:] = self.FFT_t(A)
        self.dA[:] = self.Deriv(self.Aw)

        w0 = self.w0
        t = self.T_ps
        beta0 = self.beta0
        beta1 = self.beta1
        dA = self.dA
        z = self.z

        # calculate the nonlinear operator based off Alex's Matlab code
        chi = self.gamma_function(z) * self.gbeam_approx(z)

        phase = 1j * w0 * t - 1j * (beta0 - beta1 * w0) * z
        exp_neg_phi = np.exp(-phase)
        exp_phi = np.exp(phase)
        exp_neg_phi = fftshift(exp_neg_phi)
        exp_phi = fftshift(exp_phi)

        nLterm = (2 * A + (2j / w0) * dA) * exp_neg_phi + \
                 num_over_denom((4j / w0) * np.real(np.conj(A) * dA) * exp_phi, A)

        return 1j * chi * nLterm


# I liked the change to the integrate_over_dz and propagate function in the Three Wave Mixing class
# and so here is the Four Wave Mixing class with the same alterations, I'm not changing the Nonlinear Operator
# and so the simulation here is still for SPM
class FiberFourWaveMixing(SSFM):
    def __init__(self):
        super().__init__()

    def integrate_over_dz(self, zstart, zfinish, direction=1):
        PPLNThreeWaveMixing.integrate_over_dz(self, zstart, zfinish, direction)

    def propagate(self, pulse_in, fiber, n_steps, output_power=None, reload_fiber_each_step=False):
        return PPLNThreeWaveMixing.propagate(self, pulse_in, fiber, n_steps, output_power,
                                             reload_fiber_each_step)


# This is the FiberInstance class with the added ability to set the dispersion based on D and Dprime
# instead of having to pass in betas
class Fiber(FiberInstance):
    def __init__(self):
        super().__init__()

    # given D and Dprime calculate beta2 and beta3
    def convert_D_to_beta(self, lamda_nm, D_psnmkm, D_psnm2km):
        D = D_psnmkm * 1e9 * 1e-3  # -> ps/m
        Dprime = D_psnm2km * 1e18 * 1e-3  # -> ps/m^2
        c = self.c_mks * 1e-12
        lamda = lamda_nm * 1e-9  # nm -> m

        beta2 = -lamda ** 2 * D / (2 * np.pi * c)
        beta3 = lamda ** 3 * (2 * D + Dprime * lamda) / (4 * c ** 2 * np.pi ** 2)
        return beta2, beta3

    # GVD is in terms of ps^n/m
    # and D is in terms of ps/nmkm
    def generate_fiber(self, length, center_wl_nm, betas_Ds, gamma_W_m, gain=0,
                       dispersion_format='GVD', label='Simple Fiber'):

        self.length = length
        self.fiberspecs = {'dispersion_format': 'GVD'}
        self.fibertype = label
        if gain == 0:
            self.fiberspecs["is_gain"] = False
        else:
            self.fiberspecs["is_gain"] = True
        self.gain = gain
        # The following line signals get_gain to use a flat gain spectrum
        self.fiberspecs['gain_x_data'] = None

        self.center_wavelength = center_wl_nm
        self.gamma = gamma_W_m

        # I added these
        if dispersion_format == 'GVD':
            self.betas = np.copy(np.array(betas_Ds))
        elif dispersion_format == 'D':
            self.betas = np.array(self.convert_D_to_beta(center_wl_nm, betas_Ds[0], betas_Ds[1]))
        else:
            raise ValueError("dispersion format should either be GVD or D")


def get_bandpass_filter(ref_pulse, ll_um, ul_um):
    return np.where(np.logical_and(ref_pulse.wl_um >= ll_um, ref_pulse.wl_um <= ul_um), 1, 0)


def power_in_window(pulse, AW, ll_um, ul_um, frep_MHz):
    h = get_bandpass_filter(pulse, ll_um, ul_um)
    filtered = AW * h
    AT = fftshift(fft(fftshift(filtered, axes=1), axis=1), axes=1)
    return simps(abs(AT) ** 2, axis=1) * pulse.dT_mks * frep_MHz * 1e6


def get_2d_evolv(aw2d):
    norm = np.max(abs(aw2d) ** 2, axis=1)
    toplot = abs(aw2d) ** 2
    toplot = (toplot.T / norm).T
    return toplot
