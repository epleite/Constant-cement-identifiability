# E1 methods and definitions

## Target coordinates

The nominal target vector is

\[
\theta=(v,\ell)=(V_{\rm cem},\ln C_n),
\]

with fixed RPIA scales \(s_v=0.015\) and \(s_\ell=0.20\). The forward Jacobian columns returned by the frozen code are derivatives with respect to the standardized coordinates.

Two slope estimates are retained throughout:

\[
\beta_{\rm eig}
=-\frac{s_\ell}{s_v}\frac{q_{-,\ell}}{q_{-,v}},
\]

where \(q_-\) is the weak eigenvector, and

\[
\beta_{\rm proj}
=\frac{s_\ell}{s_v}\frac{G_{v\ell}}{G_{\ell\ell}}.
\]

They coincide only in the rank-one limit. The original empirical value 20.3 is an eigen-direction slope.

## Nuisance adjustment

For standardized target and nuisance Jacobians \(J_\theta\) and \(J_\eta\), E1 uses the exact RPIA Schur complement

\[
G_{\rm adj}=J_\theta^TJ_\theta-
J_\theta^TJ_\eta
(J_\eta^TJ_\eta+I)^{-1}
J_\eta^TJ_\theta.
\]

The identity matrix is the unit Gaussian prior in standardized nuisance coordinates. E1 reports the minimum eigenvalue, spectral ratio \(\lambda_{\min}/\lambda_{\max}\), and angular-independence factor \(1-\rho^2\).

## Contact endpoint

For Scheme 1,

\[
d\ln a_c=\frac{dv}{4v}-\frac14d\ell.
\]

Writing \(m_N=d\ln S_N/d\ln a_c\) and defining the corresponding weighted exponent \(m_G\) for the shear endpoint gives

\[
\beta_{K_b}=\frac{m_N}{v(4-m_N)},
\qquad
\beta_{G_b}=\frac{m_G}{v(4-m_G)}.
\]

The endpoint Jacobian determinant is

\[
\det J_b=\frac{m_N-m_G}{4v},
\]

so endpoint separability is generated only by the small difference between normal and tangential contact responses.

## Contact-versus-HS decomposition

For any dry observable \(\chi=F(a,\phi_b,C_n)\), define

\[
A=\partial_{\ln a}\ln\chi,\quad
B=\partial_{\phi_b}\ln\chi,\quad
C=\left.\partial_{\ln C_n}\ln\chi\right|_{a,\phi_b}.
\]

Then

\[
\beta_\chi=
\frac{A/(4V_{\rm cem})-B}{C-A/4}.
\]

E1 evaluates these derivatives by a latent-variable implementation in which \(a\), \(\phi_b\), and \(C_n\) can be perturbed independently. The two reported contributions are

\[
\beta_{\rm contact}=\frac{A/(4V_{\rm cem})}{C-A/4},
\qquad
\beta_{\rm HS}=\frac{-B}{C-A/4}.
\]

## Finite ridge

The local exponential coordinate predicts

\[
\ln C_n(v)=\ln C_{n,0}-\beta_0(v-v_0).
\]

For a variable slope, E1 integrates

\[
\frac{d\ln C_n}{dv}=-\beta(v,\ln C_n).
\]

The raw integrated curve is compared with a nonlinear structural profile using pseudo-data \(f(\theta_0,0)\). The adjusted curve is compared with a second structural profile in which all nuisance parameters are jointly optimized with the same unit-Gaussian penalty used in the Schur complement.

The observed-data profile is independently re-optimized with the transfer-aware covariance. It does not reuse the original RPIA profile baseline, whose calibration and profile weights differed.

## Physics-factored trajectory coordinate

In whitened observable space, the projection slope can be decomposed linearly:

\[
\beta_{\rm proj}
=\frac{j_\ell^TPj_v}{j_\ell^TPj_\ell}
=\beta_a+\beta_{\rm HS},
\]

where \(P=I\) for the raw metric and

\[
P=I-J_\eta(J_\eta^TJ_\eta+I)^{-1}J_\eta^T
\]

for the nuisance-adjusted metric. The two target contributions are isolated in latent coordinates:

\[
j_v^{(a)}=\frac{1}{4v}\partial_{\ln a}d,
\qquad
j_v^{({\rm HS})}=-\partial_{\phi_b}d.
\]

At a reference state \(v_0\), define

\[
A=v_0\beta_a(v_0),
\qquad
\Gamma=-\phi_{b,0}\beta_{\rm HS}(v_0).
\]

Because \(\partial_v\ln T=1/\phi_b\), the natural factorization is

\[
\beta(v)\simeq\frac{A}{v}-\frac{\Gamma}{\phi_c-v}.
\]

Integrating \(d\ell/dv=-\beta(v)\) gives the centered physics-factored coordinate

\[
q_{\rm phys}
=\ln\frac{C_n}{C_{n,0}}
+A\ln\frac{v}{v_0}
+\Gamma\ln\frac{\phi_c-v}{\phi_c-v_0},
\]

or, equivalently,

\[
\chi_{\rm phys}
=\frac{C_n}{C_{n,0}}
\left(\frac{v}{v_0}\right)^A
\left(\frac{\phi_c-v}{\phi_c-v_0}\right)^\Gamma.
\]

The simpler form with \(-B(v-v_0)\), where \(B=\Gamma/\phi_{b,0}\), is the local linearization of the logarithmic \(\phi_b\) term. Both are retained in the output tables. The logarithmic form is used as the primary coordinate because it follows the exact HS-chain variable.

This is tested as a finite approximate invariant, not assumed to be a universal material coordinate. Its coefficients depend on the operating point, observable weighting, nuisance model, and constitutive assumptions.

## Candidate-state invariance

Along the raw structural profile, E1 normalizes and compares the local empirical exponential, contact radius, contact endpoints, HS coordinate, dry moduli, and saturated velocities. A candidate “effective contact state” can be treated as a finite physical invariant only if it remains approximately constant along the same prediction-preserving ridge. This is a stricter test than matching the local tangent at one operating point.
