### Purpose
This repository generates scenarios modeling variations in miner behavior.  The scenarios generate time-indexed vectors of the following indicators of miner behavior:
  - rb_onboard_power
  - renewal_rate
  - filplus_rate

The scenarios can be fed to the filecoin-mecha-twin module to understand the effect of these parameters on indicators such as:
  - Baseline crossing
  - Miner ROI
  - etc ...

We can generate four classes of scenarios:
1. Static
2. Curated scenario
3. Forecasting
    * MCMC forecasting
    * AI based forecasting

