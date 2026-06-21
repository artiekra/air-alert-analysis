# Ukraine Air Raid Alert Analytics

https://alert.artiekra.org

Interactive Streamlit dashboard for real-time geospatial and temporal analysis of air raid alert patterns across Ukraine.

The app uses alert logs and applies data clustering, timezone localization, and time series decomposition to visualize alert volumes, temporal kinetics (diurnal/seasonality), and geospatial wave propagation.

## Installation

1. **Clone and navigate to the project directory:**

   ```bash
   cd air-alerts-analysis
   ```

2. **Create and activate a virtual environment:**
   - **Linux/macOS:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Once the dependencies are installed and your virtual environment is active, you can launch the dashboard by running:

```bash
streamlit run main.py
```

Streamlit will start a local development server and automatically open the dashboard in your default web browser (typically at `http://localhost:8501`).

## Data Source

Data is sourced automatically on load from the [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) repository. No manual CSV downloading is required.
