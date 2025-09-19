# RecruitSmart Application Setup Guide

This guide provides instructions to set up, install dependencies, and run the RecruitSmart application.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/).
*   **Git**: Download from [git-scm.com](https://git-scm.com/downloads).

## Setup Instructions

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

Open your terminal or command prompt and clone the repository using Git:

```bash
git clone https://github.com/Kabilant87/Recruitapp_backend_POC.git
cd Recruitapp_backend_POC
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

### 3. Install Dependencies

With your virtual environment activated, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Once all dependencies are installed, you can run the application:

```bash
python app_gui.py
```

This will launch the RecruitSmart GUI application.

---

Feel free to reach out if you encounter any issues during the setup process.