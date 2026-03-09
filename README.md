# Proactive Safety Modeling for Traffic Conflict and Crash Prediction

This repository contains the code for proactive road safety modeling based on non-crash and pre-crash trajectory data.  
The project focuses on identifying traffic conflicts, modeling extreme events using EVT and machine learning.

The workflow includes trajectory processing, surrogate safety indicator computation, extreme value modeling, crash probability estimation, and visualization.

---

## Dataset

The dataset used in this project is available on Zenodo:

https://doi.org/10.5281/zenodo.18920439

The dataset includes processed vehicle trajectory data and surrogate safety indicators used for traffic conflict and crash modeling.

Example files:

- `trajectories_with_angle(april_may_june).csv`
- `ttc_lttb_final_split_by_type_yaw(april_may_june).csv`
- `case_1.csv`

---

## Project Structure
## Code Description

This section describes the purpose of each script in the repository.

### Data Processing

**1.threshold_selection.py**
Implements threshold selection for Extreme Value Theory (EVT) modeling.
This step identifies an appropriate threshold for Peak Over Threshold (POT) analysis.

**2.convert_trajectory_xml_to_csv.py**
Converts vehicle trajectory data generated from microscopic traffic simulation (SUMO XML format) into CSV format for subsequent analysis.

---

### Surrogate Safety Indicators

**3.conflict_indicators_calculation.py**
Calculates surrogate safety indicators from trajectory data, including metrics such as TTC (Time to Collision) and LTTB.

---

### Conflict Visualization

**4.1.conflict_visualization.py**
Visualizes traffic conflicts based on calculated surrogate safety indicators.

**4.2.conflict_visualization_different_lanes.py**
Analyzes and visualizes conflict patterns across different lanes.

---

### Conflict and Crash Modeling

**5.conflict_crash.py**
Analyzes the relationship between traffic conflicts and crash occurrences.

**6.lttb_crash_model.py**
Applies Extreme Value Theory (EVT) to model crash risk based on surrogate safety indicators.

**6.3(right_version).py**
Improved version of the EVT-based crash modeling implementation with corrected statistical procedures.

---

### Crash Prediction

**7.crash_prediction.py**
Estimates crash probability based on surrogate safety indicators and EVT modeling results.

---

### Probability Analysis and Visualization

**8.probability_visualization.py**
Visualizes probability distributions derived from EVT modeling.

**8.1(right_version).py**
Corrected version of probability visualization and probability distribution analysis.

**crash_probability_visualization.py**
Generates visualizations for crash probability results and model outputs.

---

### Machine Learning Modeling

**9.conflict_crash_modeling(Random forest).ipynb**
Implements a Random Forest model to predict crash occurence based on surrogate safety indicators.

**random forest.py**
Implements a Random Forest model to predict crash occurence based on surrogate safety indicators.

---

### Data Collection

**data_collection_TOMTOM.py**
Collects real-time traffic data using the TomTom API for use in traffic modeling and simulation.


---

### Raw Trajectory Data

**veh_trajectories.xml**
Raw vehicle trajectory data generated from traffic simulation.

**vehicle_trajectories.xml**
Detailed vehicle trajectory dataset used for traffic conflict analysis.

