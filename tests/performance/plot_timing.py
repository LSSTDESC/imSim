"""
Make plots of the run_timing_tests.py output.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

sky_bg_data = np.recfromtxt('sky_bg_timing.txt',
                            names='nxy fast slow fast_nphot slow_nphot'.split())
ratio = sky_bg_data['slow']/sky_bg_data['fast']
plt.figure()
plt.errorbar(sky_bg_data['nxy']**2, ratio, fmt='.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# pixels')
plt.ylabel('unbundled/bundled')
plt.title('ratio of sky background cpu times')
plt.savefig('sky_bg_slow_fast_scaling.png')

plt.figure()
plt.errorbar(sky_bg_data['nxy']**2, sky_bg_data['fast'], fmt='.',
             label='bundled photons (x20)')
plt.errorbar(sky_bg_data['nxy']**2, sky_bg_data['slow'], fmt='.',
             label='unbundled photons')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# pixels')
plt.ylabel('cpu time (s, cori-Haswell)')
axis = list(plt.axis())
axis[2] = 0
plt.axis(axis)
plt.legend(loc=0)
plt.title('sky background cpu times')
plt.savefig('sky_bg_fast_execution_30s_rband.png')

star_data = np.recfromtxt('star_timing.txt',
                          names='nrecalc flux fast slow'.split())
nrecalc = sorted(set(star_data['nrecalc']))
star_ratio = star_data['slow']/star_data['fast']
plt.figure()
for nr in nrecalc:
    index = np.where(star_data['nrecalc'] == nr)
    if nr == nrecalc[0]:
        label = 'nrecalc=%i, w/ sensor' % nr
    else:
        label = '%i, w/ sensor' % nr
    artists = plt.errorbar(star_data['flux'][index], star_data['slow'][index],
                           fmt='.', label=label)
    plt.errorbar(star_data['flux'][index], star_data['fast'][index],
                 fmt='v', label='%i, no sensor' % nr,
                 color=artists[0].get_color())
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('flux (e-)')
    plt.ylabel('cpu time (s, cori-Haswell)')
plt.legend(loc=0)
plt.title('single star, w/ and w/out sensor model enabled')
plt.savefig('star_timing_nrecalc.png')

plt.figure()
for nr in nrecalc:
    index = np.where(star_data['nrecalc'] == nr)
    if nr == nrecalc[0]:
        label = 'nrecalc=%i' % nr
    else:
        label = '%i' % nr
    plt.errorbar(star_data['flux'][index],
                 star_data['slow'][index]/star_data['fast'][index],
                 fmt='.', label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('flux (e-)')
    plt.ylabel('sensor/no_sensor')
plt.legend(loc=0)
plt.title('single star, ratio of cpu times')
plt.savefig('star_timing_ratio.png')
