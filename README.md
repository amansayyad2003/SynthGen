# 🛡️ SynthGen

### Synthetic Data Generator with Fraud Pattern Simulation

SynthGen is a Python-based synthetic data generation application built using **Streamlit**. It allows users to generate realistic synthetic datasets from dynamic schemas and simulate fraud patterns by configuring legitimate and fraudulent records.

The application helps users understand how fraud scenarios affect generated datasets by applying configurable fraud patterns and explaining the modifications introduced in fraudulent records.

---

# 🚀 Features

* 📄 **Dynamic Schema Support**

  * Upload or paste a JSON schema to define the structure of synthetic data.
  * Generate synthetic data based on user-defined schemas.

* ⚖️ **Fraud Simulation**

  * Configure the percentage of fraudulent records.
  * Generate datasets containing both legitimate and fraudulent entries.

* 🔍 **Fraud Pattern Detection**

  * Detect possible fraud patterns from the uploaded schema.
  * Select fraud scenarios to apply during synthetic data generation.

* 🧪 **Synthetic Data Generation**

  * Generate synthetic records with a single click.
  * View generated records in an interactive table.
  * Inspect applied fraud patterns for each fraudulent record.

* 💡 **Explain Fake Data**

  * Understand how fraudulent records were generated.
  * Provides explanations of applied fraud patterns.

* 📥 **Data Export**

  * Download generated datasets as:

    * CSV
    * JSON (ZIP compressed)

---

# 🛠️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/synthgen.git
cd synthgen
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment to isolate project dependencies.

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

SynthGen uses the following Python packages:

| Package         | Purpose                                 |
| --------------- | --------------------------------------- |
| `streamlit`     | Web interface and application UI        |
| `pandas`        | Data processing and table visualization |
| `numpy`         | Numerical operations                    |
| `faker`         | Realistic synthetic data generation     |
| `groq`          | Fraud pattern analysis and simulation   |
| `python-dotenv` | Environment variable management         |

Python standard library modules used:

* `json`
* `os`
* `io`
* `zipfile`
* `time`
* `re`
* `random`
* `datetime`
* `uuid`

These modules do not require separate installation.

---

# ⚙️ Configuration

SynthGen requires environment variables for external service configuration.

## 1. Create `.env` File

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add the required configuration:

```env
GROQ_API_KEY=<your_groq_api_key>
```

Replace:

```
<your_groq_api_key>
```

with your actual Groq API key.

---

## 2. Verify Project Structure

Ensure your project directory looks like:

```
SynthGen/
│
├── synthgen_app.py
├── requirements.txt
├── .env
└── README.md
```

---

# ▶️ Running SynthGen

After completing the installation and configuration steps, start the Streamlit application.

## Start Application

Run:

```bash
streamlit run synthgen_app.py
```

Streamlit will start the application server.

By default, the application will be available at:

```
http://localhost:8501
```

Open this URL in your browser.

---

## Running on a Custom Port

If port `8501` is already occupied, specify another port:

```bash
streamlit run synthgen_app.py --server.port 8080
```

The application will then be available at:

```
http://localhost:8080
```

---

# 📖 Using SynthGen

Follow these steps after launching the application:

### 1. Upload Schema

* Upload a JSON schema file or paste the schema directly into the application.
* The schema defines the structure of generated synthetic records.

Example:

```json
{
  "customer_id": "string",
  "transaction_amount": "number",
  "merchant": "string",
  "transaction_date": "date"
}
```

---

### 2. Configure Fraud Simulation

Configure:

* Number of records to generate
* Fraud percentage
* Fraud patterns to apply

Example:

```
Total Records: 1000
Fraud Percentage: 10%
```

This generates:

```
900 legitimate records
100 fraudulent records
```

---

### 3. Generate Synthetic Data

Click the **Generate Data** button.

The application will:

* Create synthetic records.
* Apply selected fraud patterns.
* Mark fraudulent entries.
* Store fraud explanations.

---

### 4. Analyze Generated Data

You can:

* View generated records in an interactive table.
* Identify fraudulent records.
* Inspect applied fraud patterns.
* Use **Explain Fake Data** to understand modifications.

---

### 5. Download Results

Generated datasets can be exported as:

* CSV file
* JSON ZIP file
