import numpy as np
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

data = np.loadtxt("../gaia_bar.csv", delimiter=",", skiprows=1)  # Load CSV
ra, dec, plx, pmra, pmdec, rv = data.T  # Unpack columns

coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=1000/plx*u.pc,  # Parallax in mas -> pc
                  pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                  radial_velocity=rv*u.km/u.s, frame="icrs")
gal = coords.transform_to(Galactic)
x, y, z = gal.cartesian.xyz.value / 1000  # kpc
vx, vy, vz = coords.velocity.d_xyz.value  # km/s (approximate)

target = np.column_stack((x, y, z, vx, vy, vz))
np.save("target_bar_gaia.npy", target)  # N x 6 array

