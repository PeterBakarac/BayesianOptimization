#include <Arduino.h>
#include <BayesOpt.h>

BayesOpt optimizer;

// Example test function (you can replace with your real sensor or expensive function)
float testFunction(float x) {
  // Some arbitrary function with multiple maxima
  float val = 10.0
              + 5.0 * sin(0.5f * x)
              + 3.0 * cos(1.5f * x)
              - 0.1f * (x - 5.0f) * (x - 5.0f);

  // Maybe add small noise to mimic real measurement
  float noise = (random(-100, 101) / 100.0f) * 0.3f; // ±0.3
  return val + noise;
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("=== Bayesian Optimization Example ===");

  // Set random seed for demonstration
  randomSeed(esp_random());

  // Configure optimizer parameters
  optimizer.setParameters(
    0.2f,     // noise
    2.0f,     // length scale
    5.0f,     // sigmaF
    2.0f,     // alpha (exploration)
    0.0f,     // domainMin
    12.0f,    // domainMax
    0.5f,     // domainStep
    20        // max points
  );

  // Allocate internal memory, reset data
  optimizer.begin();

  // Optionally add a couple of seed points
  optimizer.addDataPoint(2.0, testFunction(2.0));
  optimizer.addDataPoint(8.0, testFunction(8.0));

  Serial.println("Added initial seed points.");
  Serial.println("Maximum number of iterations: " + String(optimizer.getMaxPoints()));
}

void loop() {
  // 1) Use GP + UCB to pick next candidate
  float nextX = optimizer.findNextCandidateUCB();

  // 2) Evaluate the function (or do real sensor measurements)
  float yObs = testFunction(nextX);

  // 3) Add data to the GP
  optimizer.addDataPoint(nextX, yObs);

  // Print iteration info
  int iterNum = optimizer.getNumPoints();
  Serial.print("Iteration #");
  Serial.print(iterNum);
  Serial.print("; chosen X = ");
  Serial.print(nextX, 2);
  Serial.print("; observed Y = ");
  Serial.print(yObs, 2);
  Serial.println("");

  // If we have enough data points, stop
  if (iterNum >= optimizer.getMaxPoints()) {
    Serial.println("Reached maximum data points. Stopping.");
    float bestX = optimizer.findMaxMean();
    Serial.println("Solution found at point: " + String(bestX, 2));
    while(true) { delay(1000); }
  }

  delay(200);
}
