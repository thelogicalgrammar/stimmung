/*
 * Kelly-Lochbaum vocal tract with a nasal side branch.
 *
 * Pressure-wave digital waveguide: every tube section is one sample long in
 * each direction (section length = c / fs).  Junctions are solved as N-port
 * pressure junctions, so the same code handles the 2-port oral junctions and
 * the 3-port velum junction.  Glottal flow is a smooth Rosenberg/LF-like pulse
 * with aspiration noise; lips and nostrils reflect through a one-pole lowpass.
 *
 * Output is the time derivative of (lip + nostril) volume velocity, i.e. the
 * radiated pressure up to a constant.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ULL;
static inline double urand(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (rs >> 11) * (1.0 / 9007199254740992.0);
}
static inline double nrand(void) {
    /* cheap approximately-gaussian noise */
    return (urand() + urand() + urand() + urand() - 2.0) * 1.732;
}

/* one period of glottal flow, phase t in [0,1), tense in [0,1] */
static inline double glottal(double t, double tense) {
    double oq = 0.78 - 0.33 * tense;          /* open quotient */
    double tp = oq * (0.62 + 0.12 * tense);    /* peak position */
    if (t < tp) return 0.5 * (1.0 - cos(M_PI * t / tp));
    if (t < oq) {
        double x = (t - tp) / (oq - tp);
        return cos(0.5 * M_PI * x);
    }
    return 0.0;
}

void synth_voice(int nsamp, int hop, int nframes, int N, int NN, int vel,
                 const double *area, const double *nasal, const double *velum,
                 const double *f0, const double *amp, const double *breath,
                 const double *tense, double fs, double glot_refl,
                 double lip_a, double lip_b, double damp, double jitter,
                 unsigned long long seed, double *out)
{
    rs = seed ? seed : 88172645463325252ULL;
    double *R = calloc(N, sizeof(double)), *L = calloc(N, sizeof(double));
    double *Rn = calloc(N, sizeof(double)), *Ln = calloc(N, sizeof(double));
    double *A = calloc(N, sizeof(double));
    double *NR = calloc(NN, sizeof(double)), *NLw = calloc(NN, sizeof(double));
    double *NRn = calloc(NN, sizeof(double)), *NLn = calloc(NN, sizeof(double));
    double *An = calloc(NN, sizeof(double));
    memcpy(An, nasal, NN * sizeof(double));

    double phase = 0.0, lp_lip = 0.0, lp_nose = 0.0, prevU = 0.0;
    double nz_lp = 0.0, nz_lp2 = 0.0, nz_hp = 0.0;
    /* pitch: per-cycle jitter + fast flutter + slow wander; amplitude: shimmer + swell */
    double cyc = 1.0, flut = 0.0, wand = 0.0, shim = 1.0, swell = 0.0;
    const double a_fl = exp(-1.0 / (0.04 * fs)), a_wa = exp(-1.0 / (0.6 * fs));
    const double a_sw = exp(-1.0 / (0.35 * fs));
    const double k_fl = 0.0015 * sqrt(1 - a_fl * a_fl), k_wa = 0.0025 * sqrt(1 - a_wa * a_wa);
    const double k_sw = 0.07 * sqrt(1 - a_sw * a_sw);

    for (int n = 0; n < nsamp; n++) {
        int fi = n / hop;
        double fr = (double)(n % hop) / hop;
        int f1 = fi + 1 < nframes ? fi + 1 : nframes - 1;
        if (fi >= nframes) { fi = nframes - 1; f1 = fi; fr = 0; }
        const double *a0 = area + (size_t)fi * N, *a1 = area + (size_t)f1 * N;
        for (int i = 0; i < N; i++) A[i] = a0[i] + fr * (a1[i] - a0[i]);
        double vA = velum[fi] + fr * (velum[f1] - velum[fi]);
        double F0 = f0[fi] + fr * (f0[f1] - f0[fi]);
        double AMP = amp[fi] + fr * (amp[f1] - amp[fi]);
        double BR = breath[fi] + fr * (breath[f1] - breath[fi]);
        double TN = tense[fi] + fr * (tense[f1] - tense[fi]);
        An[0] = vA;

        flut = a_fl * flut + jitter * k_fl * nrand();
        wand = a_wa * wand + jitter * k_wa * nrand();
        swell = a_sw * swell + jitter * k_sw * nrand();
        phase += F0 * cyc * (1.0 + flut + wand) / fs;
        if (phase >= 1.0) {
            phase -= 1.0;
            cyc = 1.0 + jitter * 0.004 * nrand();
            shim = 1.0 + jitter * 0.05 * nrand();
        }
        double g = glottal(phase, TN);
        /* aspiration: two-pole lowpass (~1.4 kHz; radiation lifts it back up),
           highpassed, pulsed by glottal opening */
        nz_lp = 0.9 * nz_lp + 0.1 * nrand();
        nz_lp2 = 0.9 * nz_lp2 + 0.1 * nz_lp;
        double asp = nz_lp2 - nz_hp; nz_hp = 0.98 * nz_hp + 0.02 * nz_lp2;
        double Ug = AMP * shim * (1.0 + swell) * g + BR * asp * (0.15 + g);

        /* glottis: nearly closed end when folds shut, more absorbing when open */
        double gr = glot_refl + (0.97 - glot_refl) * (1.0 - g);
        Rn[0] = gr * L[0] + Ug / A[0];
        /* oral junctions */
        for (int j = 1; j < N; j++) {
            if (j == vel && vA > 1e-6) {
                double s = A[j - 1] + A[j] + vA;
                double PJ = 2.0 * (A[j - 1] * R[j - 1] + A[j] * L[j] + vA * NLw[0]) / s;
                Rn[j] = PJ - L[j];
                Ln[j - 1] = PJ - R[j - 1];
                NRn[0] = PJ - NLw[0];
            } else {
                double PJ = 2.0 * (A[j - 1] * R[j - 1] + A[j] * L[j]) / (A[j - 1] + A[j]);
                Rn[j] = PJ - L[j];
                Ln[j - 1] = PJ - R[j - 1];
                if (j == vel) NRn[0] = 0.0;
            }
        }
        /* lips */
        lp_lip = (1.0 - lip_b) * R[N - 1] + lip_b * lp_lip;
        Ln[N - 1] = -lip_a * lp_lip;
        double Ulip = A[N - 1] * (R[N - 1] - Ln[N - 1]);

        /* nasal tract */
        for (int j = 1; j < NN; j++) {
            double PJ = 2.0 * (An[j - 1] * NR[j - 1] + An[j] * NLw[j]) / (An[j - 1] + An[j] + 1e-12);
            NRn[j] = PJ - NLw[j];
            NLn[j - 1] = PJ - NR[j - 1];
        }
        lp_nose = (1.0 - lip_b) * NR[NN - 1] + lip_b * lp_nose;
        NLn[NN - 1] = -lip_a * lp_nose;
        double Unose = An[NN - 1] * (NR[NN - 1] - NLn[NN - 1]);

        for (int i = 0; i < N; i++) { R[i] = damp * Rn[i]; L[i] = damp * Ln[i]; }
        for (int i = 0; i < NN; i++) { NR[i] = damp * NRn[i]; NLw[i] = damp * NLn[i]; }

        double U = Ulip + Unose;
        out[n] = U - prevU;
        prevU = U;
    }
    free(R); free(L); free(Rn); free(Ln); free(A);
    free(NR); free(NLw); free(NRn); free(NLn); free(An);
}
