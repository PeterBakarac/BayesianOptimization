#include "BayesOpt.h"
#include <math.h>
#include <stdlib.h> // for malloc() and free() functions

BayesOpt::BayesOpt() {
    // Set some default values
    _noise        = 0.2f;
    _lengthScale  = 2.0f;
    _sigmaF       = 5.0f;
    _alpha        = 2.0f;
    _domainMin    = 0.0f;
    _domainMax    = 10.0f;
    _domainStep   = 0.5f;
    _maxPoints    = MAX_POINTS_DEFAULT;
    N             = 0;
    X             = nullptr;
    Y             = nullptr;
    K_inv         = nullptr;
}

void BayesOpt::setParameters(
    float noise,
    float lengthScale,
    float sigmaF,
    float alpha,
    float domainMin,
    float domainMax,
    float domainStep,
    int   maxPoints
) {
    _noise       = noise;
    _lengthScale = lengthScale;
    _sigmaF      = sigmaF;
    _alpha       = alpha;
    _domainMin   = domainMin;
    _domainMax   = domainMax;
    _domainStep  = domainStep;
    _maxPoints   = maxPoints;
}

// Optionally call this in your setup(), if you want to do memory allocation or resets
void BayesOpt::begin() {
    if (X) free(X);
    if (Y) free(Y);
    if (K_inv) free(K_inv);

    X = (float*)malloc(_maxPoints * sizeof(float));
    Y = (float*)malloc(_maxPoints * sizeof(float));
    K_inv = (float*)malloc(_maxPoints * _maxPoints * sizeof(float));

    // Initialize
    N = 0;
}

void BayesOpt::setDomain(float minVal, float maxVal, float stepVal) {
    _domainMin  = minVal;
    _domainMax  = maxVal;
    _domainStep = stepVal;
}

// Add a new data point (x, y) to the arrays
void BayesOpt::addDataPoint(float x, float y) {
    if (N < _maxPoints) {
        X[N] = x;
        Y[N] = y;
        N++;
    } else {
        Serial.println("WARNING: Reached maximum number of data points; ignoring new data!");
    }
}

// Build the NxN covariance matrix K[X,X] and invert it
void BayesOpt::buildCovarianceAndInvert() {
    if (N == 0) return;

    float *K = (float*)malloc(N * N * sizeof(float));

    // Build K
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            K[i*N + j] = rbfKernel(X[i], X[j]);
        }
    }

    // Add noise to diagonal to improve the numerical stability
    for (int i = 0; i < N; i++) {
        K[i*N + i] += (_noise * _noise);
    }

    // Copy K into K_inv (we invert in place)
    for (int i = 0; i < N*N; i++) {
        K_inv[i] = K[i];
    }

    // Invert
    invertMatrix(K_inv, N);

    free(K);
}

// Compute posterior mean & variance at candidate x*
// m[x*] = K[x*,X]K[X,X]^(-1)f
// phi[x*] = K[x*,x*] - K[x*,X]K[X,X]^-1 K[X,x*]
void BayesOpt::computePosterior(float candidate, float &meanOut, float &varOut) {
    if (N == 0) {
        // No data -> wide open prior
        meanOut = 0.0f;
        varOut  = 1e6f;
        return;
    }

    // ---- mean ----
    // kVec = K[x*,X]
    float *kVec = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        kVec[i] = rbfKernel(candidate, X[i]);
    }

    // temp = K[X,X]^(-1)f
    float *temp = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += K_inv[i*N + j] * Y[j];
        }
        temp[i] = sum;
    }

    // m[x*] = K[x*,X]K[X,X]^(-1)f
    // meanCand = kVec^T * temp
    float meanCand = 0.0f;
    for (int i = 0; i < N; i++) {
        meanCand += kVec[i] * temp[i];
    }

    // ---- variance ----
    // phi[x*] = K[x*,x*] - K[x*,X]K[X,X]^-1 K[X,x*]
    // var = k(candidate,candidate) - kVec^T * K_inv * kVec

    // K[X,X]^-1 K[X,x*]
    // temp2 = K_inv * kVec
    float *temp2 = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += K_inv[i*N + j] * kVec[j];
        }
        temp2[i] = sum;
    }

    // kck = K[x*,X]K[X,X]^-1 K[X,x*]
    float kck = 0.0f;
    for (int i = 0; i < N; i++) {
        kck += kVec[i] * temp2[i];
    }

    // varCand = K[x*,x*] - K[x*,X]K[X,X]^-1 K[X,x*]
    float varCand = rbfKernel(candidate, candidate) - kck;
    if (varCand < 1e-9) varCand = 1e-9; // numeric stability

    meanOut = meanCand;
    varOut  = varCand;

    // free the memory of temporary variables
    free(kVec);
    free(temp);
    free(temp2);
}

// Find next x using UCB across [domainMin, domainMax] stepping by _domainStep
float BayesOpt::findNextCandidateUCB() {
    // Rebuild covariance for fresh data
    buildCovarianceAndInvert();

    float bestUCB = -1e9f;
    float bestX   = _domainMin;

    // Evaluate discrete candidates
    for (float cand = _domainMin; cand <= (_domainMax + 1e-5f); cand += _domainStep) {
        float m, v;
        computePosterior(cand, m, v);
        float stdCand = sqrtf(v);
        float ucb = m + _alpha * stdCand;
        if (ucb > bestUCB) {
            bestUCB = ucb;
            bestX   = cand;
        }
    }

    return bestX;
}

// Evaluate the solution of The Bayesian Optimization
float BayesOpt::findMaxMean() {
    // Rebuild or ensure we have an up-to-date inverse covariance matrix
    buildCovarianceAndInvert();

    float bestMean = -1e9f;
    float bestX    = _domainMin;

    // Scan the domain in discrete steps
    for (float cand = _domainMin; cand <= _domainMax + 1e-5f; cand += _domainStep) {
        float m, v;
        computePosterior(cand, m, v);
        if (m > bestMean) {
            bestMean = m;
            bestX    = cand;
        }
    }
    return bestX;
}

// RBF Kernel (“Radial Basis Function kernel”, refers to the squared exponential (SE) kernel or Gaussian kernel)
float BayesOpt::rbfKernel(float x1, float x2) {
    float diff = x1 - x2;
    float val  = expf(-0.5f * (diff * diff) / (_lengthScale * _lengthScale)) * (_sigmaF * _sigmaF);
    return val;
}

// Naive matrix inversion via Gauss-Jordan
void BayesOpt::invertMatrix(float *M, int n) {
    // Create identity matrix
    float *I = (float*)malloc(n * n * sizeof(float));
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            I[r*n + c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    // Gauss-Jordan
    for (int c = 0; c < n; c++) {
        // Pivot selection
        float maxVal = fabs(M[c*n + c]);
        int pivot = c;
        for (int r = c+1; r < n; r++) {
            float v = fabs(M[r*n + c]);
            if (v > maxVal) {
                maxVal = v;
                pivot = r;
            }
        }

        // Swap pivot row if needed
        if (pivot != c) {
            for (int cc = 0; cc < n; cc++) {
                float tmp = M[c*n + cc];
                M[c*n + cc] = M[pivot*n + cc];
                M[pivot*n + cc] = tmp;

                tmp = I[c*n + cc];
                I[c*n + cc] = I[pivot*n + cc];
                I[pivot*n + cc] = tmp;
            }
        }

        // Scale pivot row
        float diagVal = M[c*n + c];
        if (fabs(diagVal) < 1e-12) {
            Serial.println("WARNING: Covariance matrix is near-singular!");
            free(I);
            return;
        }
        for (int cc = 0; cc < n; cc++) {
            M[c*n + cc] /= diagVal;
            I[c*n + cc] /= diagVal;
        }

        // Eliminate column c in other rows
        for (int r = 0; r < n; r++) {
            if (r != c) {
                float factor = M[r*n + c];
                for (int cc = 0; cc < n; cc++) {
                    M[r*n + cc] -= factor * M[c*n + cc];
                    I[r*n + cc] -= factor * I[c*n + cc];
                }
            }
        }
    }
    // M now is identity, I is M^-1
    // Copy back
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            M[r*n + c] = I[r*n + c];
        }
    }
    free(I);
}
