# Bayesian Optimization Arduino Library

**Bayesian Optimization** is a powerful strategy for optimizing expensive or unknown functions, utilizing a Gaussian Process (GP) to model the function and select promising points to sample by using Upper confidence bound acquisition function. This library provides a **1D GP-based Bayesian Optimization** implementation for Arduino boards, including the ESP32 family.

---

## Features

- **1D Gaussian Process** with an **RBF (Radial Basis Function)** kernel.  
- Support for **Bayesian Optimization** using **Upper Confidence Bound (UCB)** acquisition.  
- Easily configured hyperparameters:  
  - Noise term  
  - Length scale (\(\ell\))  
  - Signal variance (\(\sigma_f\))  
  - Exploration factor (\(\alpha\))  
- Simple matrix inversion (Gauss-Jordan) for small datasets.  
- Designed for microcontrollers like **ESP32**, **ESP8266**, or standard **Arduino** boards.  
- Supports discrete scanning of a user-defined domain (e.g., `[domainMin, domainMax]` with increments).
