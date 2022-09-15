# Monte Carlo

Source: [ntrs.nasa.gov/citations/19990094899](https://ntrs.nasa.gov/citations/19990094899)

## Statistical Aproach

Ideal rays / no diffraction: $\Delta \xi > 10$.

$$ \Delta \xi = a \left(\frac{2}{\lambda z}\right)^{\frac{1}{2}} $$

z: Distance aperture - observation screen

a: slit width

$\lambda$: Wave length

$$ p(\tan(\phi_d)) = \frac{1}{\sqrt{2\pi} \phi^\ast} \exp\left[-\frac{1}{2} \left(\frac{\tan(\phi_d)}{\phi^\ast}\right)^2\right] $$

Probability for the diffraction angle $\phi_d$ in terms of

$$ \phi^* = \tan^{-1}\left( \frac{1}{2k \delta} \right), $$
where
$k = \frac{2\pi}{\lambda}$
and $\delta$ is the distance to the aperture edge.

Original thesis: 4 different implementation:
* Sum the resulting angles $\phi_d$ over edges (upper / lower)
* Duplicate ray on entrance, neglect one of the 2 edges for each ray
* $\delta := \min(\delta_+, \delta_-)$
* $\delta := \max(\delta_+, \delta_-)$

Conclusion: sum approach best

Our approach:

Apperture is projection of the spider to the pupil plane.
$\delta$ is the distance to the boundary of that projection.

## Model 2

$$
E_{eb} = K(\phi_i, \phi_d) \frac{\sin(\beta)}{\beta},
\beta = 2\ell\frac{\pi}{\lambda}
$$

$\ell$: path length (?)

$K$: obliquity factor

Huygens: $K(\phi_i, \phi_d) = 1$

Kirchhoff: $K(\phi_i, \phi_d) = \frac{1}{2}(\cos(\phi_i)+\cos(\phi_d))$

Miller: $K(\phi_d) = \cos(\phi_d)$

$$
I_{bin} = \left[\sum E_{eb}(\mathrm{bin})\right]^2
$$

# Fourier Optics / presampling

## Fresnel approximation

$$
E(x, y) = \frac{1}{i\lambda z} e^{ikz} e^{i\frac{kz}{2z}(x^2+y^2)}
\int\int_\Sigma E_0(x', y') e^{i\frac{kz}{2z}(xx'+yy')}dx'dy'.
$$
