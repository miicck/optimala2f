import os
import random
import time
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE, DEVNULL
from typing import Tuple, List
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

A2TC_EXE = os.path.join(os.path.dirname(__file__), "a2tc", "bin", "a2tc")
K_TO_THZ = 0.02083661330386207
THZ_TO_K = 1.0 / K_TO_THZ


def normalize(omega: np.ndarray, a2f: np.ndarray, to_lambda: float = 1.0) -> np.ndarray:
    i_safe = omega > 1e-5
    lam = np.trapz(a2f[i_safe] / omega[i_safe], x=omega[i_safe]) * 2
    return to_lambda * a2f / lam


def run_a2tc(omega: np.ndarray, a2f: np.ndarray, mu_star: float = 0.125) -> np.ndarray:
    in_pipe = "# E(THz) 0.00\n"
    for w, a in zip(omega, a2f):
        in_pipe += f"{w} {a}\n"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    p = Popen([A2TC_EXE, "-q", "-force", "-precision", "-nsig", "1", "-mustar", str(mu_star)],
              stdin=PIPE, stdout=PIPE, stderr=DEVNULL, env=env)
    output = p.communicate(input=in_pipe.encode("utf-8"))[0].decode()
    data = [float(x) for x in output.split()]
    return np.array(data)


def calculate_lambda_tc(omega: np.ndarray, a2f: np.ndarray, mu_star: float = 0.125) -> Tuple[float, float]:
    d = run_a2tc(omega, a2f, mu_star=mu_star)
    return d[0], d[-2]


def calculate_lambda_wlog_tc(omega: np.ndarray, a2f: np.ndarray, mu_star: float = 0.125) -> Tuple[float, float]:
    d = run_a2tc(omega, a2f, mu_star=mu_star)
    return d[0], d[1], d[-2]


def a2f_guassian_peak(peak_temperature: float,
                      lambda_value: float = 1.0,
                      n_points: int = 1024,
                      frac_width: float = 0.01) -> Tuple[
    np.ndarray, np.ndarray]:
    omega = np.linspace(0, peak_temperature * 2, n_points)
    a2f = np.exp(-((omega - peak_temperature) / (frac_width * peak_temperature)) ** 2)

    omega *= K_TO_THZ
    a2f = normalize(omega, a2f, to_lambda=lambda_value)

    return omega, a2f


def a2f_from_dat(dat_file: str) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    with open(dat_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip headers
            data.append([float(x) for x in line.split()])

    data = np.array(data).T
    return data[0], data[1:]


def a2f_h3s() -> Tuple[np.ndarray, np.ndarray]:
    w, a = a2f_from_dat(os.path.join(os.path.dirname(__file__), "a2tc", "examples", "H3S.alpha2F.dat"))
    return w, a[5]


def tc_of_guassian_peak(peak_temperature: float, lambda_value: float = 1.0):
    lam, tc = calculate_lambda_tc(*a2f_guassian_peak(peak_temperature, lambda_value=lambda_value))
    print(lam, lambda_value)
    assert abs(lam - lambda_value) < 0.01
    return tc


def guassian_peak_plot():
    peak_t = np.linspace(100, 20000, 10)
    for lam in [0.5, 1.0, 1.5, 2.0]:
        with Pool(cpu_count()) as p:
            tcs = p.starmap(tc_of_guassian_peak, [[t, lam] for t in peak_t])
        gradient = tcs[-1] / peak_t[-1]
        plt.plot(peak_t, tcs, label=f"Lambda = {lam:.1f} (gradient = {gradient:.3f})")
    plt.xlabel(r"Location of Guassian peak in $\alpha^2F(\omega)$ (K)")
    plt.ylabel(r"$T_c$ from solution of" + "\n" + r"Eliashberg equations ($\mu*$ = 0.125) (K)")
    plt.legend()
    plt.show()


class A2FOptimizer(ABC):

    def __init__(self, omega: np.ndarray, attenuation_k: float = 300):
        self._omega = omega
        self._attenuation_k = attenuation_k
        self._history = []

    @abstractmethod
    def normalize_a2f(self, omega: np.ndarray, a2f: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def sqrt_to_a2f(self, sqrt_a2f: np.ndarray) -> np.ndarray:
        # Enforce positive a2f
        a2f = sqrt_a2f ** 2

        # Enforce a2f that smoothly goes to zero
        x1 = (self._omega - max(self._omega)) / (self._attenuation_k * K_TO_THZ)
        x2 = (self._omega) / (self._attenuation_k * K_TO_THZ)
        att = (1 - np.exp(-x1 ** 2)) * (1 - np.exp(-x2 ** 2))

        a2f *= att

        return self.normalize_a2f(self._omega, a2f)

    def objective(self, sqrt_a2f: np.ndarray) -> float:
        a2f = self.sqrt_to_a2f(sqrt_a2f)
        lam, tc = calculate_lambda_tc(self._omega, a2f)
        return -tc

    def delta_objective(self, sqrt_a2f: np.ndarray, i: int, eps: float):
        sqrt_a2f[i] += eps
        result = self.objective(sqrt_a2f)
        sqrt_a2f[i] -= eps
        return result

    def gradient(self, sqrt_a2f: np.ndarray) -> np.ndarray:
        eps = 1e-5

        obj_0 = self.objective(sqrt_a2f)
        with Pool(12) as p:
            obj_plus = p.starmap(self.delta_objective, [[sqrt_a2f, i, eps] for i in range(len(sqrt_a2f))])
        return (np.array(obj_plus) - obj_0) / eps

    def callback(self, x: np.ndarray):
        a2f = self.sqrt_to_a2f(x)
        lam, wlog, tc = calculate_lambda_wlog_tc(self._omega, a2f)
        self._history.append([lam, tc, wlog, a2f.copy()])

        plt.figure("Tc")
        plt.clf()
        plt.plot([t for l, t, w, a in self._history])
        plt.xlabel("Iteration")
        plt.ylabel("$T_c$")

        plt.figure("lambda")
        plt.clf()
        plt.plot([l for l, t, w, a in self._history])
        plt.xlabel("Iteration")
        plt.ylabel(r"$\lambda$")

        plt.figure("omega log")
        plt.clf()
        plt.plot([w for l, t, w, a in self._history])
        plt.xlabel("Iteration")
        plt.ylabel(r"$\omega_{log}$")

        plt.figure("a2F")
        plt.clf()
        for i, (l, t, w, a) in enumerate(self._history):
            c = (i + 1) / len(self._history)
            c = (c, 1 - c, 0)
            plt.plot(self._omega * THZ_TO_K, a, color=c)

        plt.xlabel(r"$\omega$ (K)")
        plt.ylabel(r"$\alpha^2F(\omega)$")
        plt.pause(0.5)


class A2FOptimizerFixedLambda(A2FOptimizer):

    def __init__(self, omega: np.ndarray, attenuation_k: float = 300, fix_lambda: float = 1.0):
        super().__init__(omega, attenuation_k=attenuation_k)
        self._fix_lambda = fix_lambda

    def normalize_a2f(self, omega: np.ndarray, a2f: np.ndarray) -> np.ndarray:
        lam, tc = calculate_lambda_tc(omega, a2f)
        return a2f * self._fix_lambda / lam


class A2FOptimizerFixedOmegaLog(A2FOptimizer):

    def __init__(self, omega: np.ndarray, attenuation_k: float = 300, fix_omega_log_k: float = 1.0):
        super().__init__(omega, attenuation_k=attenuation_k)
        self._fix_omega_log_k = fix_omega_log_k

    def normalize_a2f(self, omega: np.ndarray, a2f: np.ndarray) -> np.ndarray:
        lam, wlog, tc = calculate_lambda_wlog_tc(omega, a2f)

        # Shift a2F(w) -> a2F(w/a)
        from scipy.interpolate import interp1d
        a2f_f = interp1d(omega, a2f, fill_value=0, bounds_error=False, kind="quadratic")
        a = self._fix_omega_log_k / wlog
        a2f = a2f_f(omega / a)

        lam_new, wlog_new, tc_new = calculate_lambda_wlog_tc(omega, a2f)

        if abs(wlog_new - self._fix_omega_log_k) > 10:
            return self.normalize_a2f(omega, a2f)

        print(wlog_new)

        return a2f


def optimize_a2f():
    max_w = 3680
    att_k = 300

    plt.ion()
    omega = np.linspace(0, (max_w + att_k) * K_TO_THZ, 256)
    a2f = np.ones_like(omega)

    sqrt_a2f = a2f ** 0.5
    a2f_opt = A2FOptimizerFixedLambda(omega, attenuation_k=att_k, fix_lambda=2.0)
    #a2f_opt = A2FOptimizerFixedOmegaLog(omega, attenuation_k=att_k, fix_omega_log_k=600)

    a2f_opt.callback(sqrt_a2f)
    minimize(a2f_opt.objective, x0=sqrt_a2f, jac=a2f_opt.gradient, callback=a2f_opt.callback)

    plt.ioff()
    plt.show()


def optimize_phonons():
    from ase import Atoms
    from ase.phonons import Phonons
    from ase.calculators.calculator import Calculator, all_changes
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.visualize import view
    from epcdata.paths import load_local_dataset
    from epcdata.database.constants import THZ_TO_CMM, EV_TO_THZ
    from epcdata.database.utils import grab_structure_ase
    from epcdata.parsing.epc_derived_quantitites import tc_allen_dynes, lambda_from_a2f, wlog_from_a2f
    from epcdata.database.masks import subsets_of_interest
    from ase.constraints import ExpCellFilter
    import periodictable as ptable

    class MyCalculator(Calculator):

        def __init__(self):
            Calculator.__init__(self)

            self.results = {}
            self.implemented_properties = [
                "energy",
                "forces",
            ]

        def calculate(self, atoms: Atoms = None, properties: List[str] = None, system_changes=all_changes):
            Calculator.calculate(self, atoms)
            self.results = {
                "energy": 0.0,
                "forces": np.random.random((len(atoms), 3))
            }

    data = subsets_of_interest(load_local_dataset())["sas_hifi"]
    i = random.randint(0, len(data["id"]) - 1)
    atoms = grab_structure_ase(data, i)

    # atoms.set_chemical_symbols(["Al"] * len(atoms))

    emt_supported = ["Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt"]
    emt_supported = [getattr(ptable, e).number for e in emt_supported]

    def closest_z(z: int) -> int:
        result = getattr(ptable, "Al").number
        for z2 in emt_supported:
            if abs(z - z2) < abs(z - result):
                result = z2
        return result

    atoms.set_atomic_numbers([closest_z(z) for z in atoms.get_atomic_numbers()])

    atoms.calc = EMT()

    BFGS(ExpCellFilter(atoms)).run()
    view(atoms)

    # Calculate phonons on grid
    N = 2
    ph = Phonons(atoms, atoms.calc, supercell=(N, N, N), delta=0.05)
    ph.run()

    # Ensure phonon cache has been written
    for k in ph.cache:
        with ph.cache.lock(k):
            pass

    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)  # Enforce acoustic sum rule
    ph.clean()

    path = atoms.cell.bandpath(npoints=64)
    bs = ph.get_band_structure(path)

    # Get DOS
    dos = ph.get_dos(kpts=(10, 10, 10)).sample_grid(npts=128, width=1e-3)

    omega = np.array(dos.get_energies())
    dos = np.array(dos.get_weights())

    min_e = min(omega[dos > 1e-6])
    max_e = max(omega[dos > 1e-6])

    plt.figure("Bands")
    bs.plot(ax=plt.gca())
    plt.ylim(min_e, max_e)

    plt.figure("DOS")
    plt.plot(omega, dos)
    plt.xlabel("omega (eV)")
    plt.ylabel("DOS")

    omega_a2f = omega.copy()

    if True:
        omega_a2f *= EV_TO_THZ
    else:
        omega_a2f /= max_e
        omega_a2f *= 3000 * K_TO_THZ

    a2f = dos.copy()
    a2f[omega_a2f < 0] = 0
    a2f[omega_a2f < 1] *= omega_a2f[omega_a2f < 1] / 1
    a2f *= omega_a2f

    a2f = normalize(omega_a2f, a2f, to_lambda=1)
    print(omega_a2f.shape, a2f.shape)

    plt.figure("A2F")

    plt.plot(omega_a2f * THZ_TO_K, a2f)
    plt.xlabel("omega (K)")
    plt.ylabel("A2F")

    plt.ion()
    plt.pause(1)

    lam, tc = np.nan, np.nan  # calculate_lambda_tc(omega_a2f, a2f)
    lam = lambda_from_a2f(omega_a2f * THZ_TO_CMM, a2f)
    omega_log = wlog_from_a2f(omega_a2f * THZ_TO_CMM, a2f)
    tc_ad = tc_allen_dynes(omega_a2f * THZ_TO_CMM, a2f, mu_star=0.125)

    print(f"lambda = {lam:.3f} omega_log = {omega_log:.3f} Tc Eli/AD = {tc:.1f}/{tc_ad:.1f} K")

    plt.ioff()
    plt.show()


def introduction_of_soft_modes():
    import matplotlib.pyplot as plt

    def plot(omega: np.ndarray, a2f: np.ndarray):
        lam, wlog, tc = calculate_lambda_wlog_tc(omega, a2f, 0.125)
        plt.plot(omega, a2f, label=f"Lambda = {lam:.3f} "
                                   f"w_log = {wlog:.3f} "
                                   f"Tc = {tc:.3f}")
        plt.xlabel(r"$\omega (THz)$")
        plt.ylabel(r"$\alpha^2F(\omega)$")
        plt.legend(loc=1)

    N_pert = 20

    plt.subplot(1 + N_pert, 1, 1)
    omega, a2f = a2f_h3s()
    plot(omega, a2f)

    for pert in range(N_pert):

        plt.subplot(1 + N_pert, 1, 2 + pert)

        a2f_pert = a2f.copy()

        for n in range(random.randint(1, 10)):
            f = np.random.random() * max(omega) / 10.0

            x = (omega - f) / (10 * K_TO_THZ)
            da2f = np.exp(-x ** 2)
            da2f /= max(da2f)
            a2f_pert += da2f * np.random.random() * 5

        plot(omega, a2f_pert)

    plt.show()


if __name__ == "__main__":
    optimize_a2f()
    # introduction_of_soft_modes()
