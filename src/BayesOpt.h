#ifndef BAYESOPT_H
#define BAYESOPT_H

#include <Arduino.h>

class BayesOpt {
public:
    BayesOpt();

    void setParameters(
        float noise,
        float lengthScale,
        float sigmaF,
        float alpha,
        float domainMin,
        float domainMax,
        float domainStep,
        int   maxPoints
    );

    void begin();

    void addDataPoint(float x, float y);

    void buildCovarianceAndInvert();

    void computePosterior(float candidate, float &meanOut, float &varOut);

    float findNextCandidateUCB();
    
    float findMaxMean();

    // Access the current number of data points
    int   getNumPoints() const { return N; }

    int getMaxPoints() const { return _maxPoints; }

    // Access or change the domain stepping, min and max domain
    void  setDomain(float minVal, float maxVal, float stepVal);

private:

    float rbfKernel(float x1, float x2);

    void  invertMatrix(float *M, int n);

    // Hyperparameters
    float _noise;
    float _lengthScale;
    float _sigmaF;
    float _alpha;

    // Discrete domain search
    float _domainMin;
    float _domainMax;
    float _domainStep;

    // Data arrays
    static const int MAX_POINTS_DEFAULT = 20;
    float *X; // dynamic or static depending on how you want to store them
    float *Y;
    int    _maxPoints;
    int    N; // number of data points

    // Covariance inverse
    float *K_inv; // store in 1D array form: size = _maxPoints * _maxPoints
};

#endif
