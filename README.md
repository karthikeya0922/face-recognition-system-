
```markdown
# 🧠 Face Recognition Attendance System

A smart attendance tracking system built using **Flask**, **OpenCV**, and **Python** that marks attendance automatically using **face recognition**.  
Simple frontend built with **HTML** and **CSS** — lightweight, fast, and functional.

---

## 🚀 Features
- 🧍‍♂️ Detects and recognizes faces in real time using OpenCV.
- 🗓️ Automatically records attendance in `.csv` files with timestamps.
- 💾 Stores daily attendance logs in the `Attendance/` folder.
- 🌐 Flask-powered web interface for user interaction.
- 🎨 Basic HTML/CSS frontend for simplicity and ease of use.

---

## 🛠️ Tech Stack
| Category | Tech |
|-----------|------|
| Backend | Python, Flask |
| Face Detection | OpenCV, Haar Cascade Classifier |
| Frontend | HTML, CSS |
| Data Storage | CSV Files |

---

## 📁 Project Structure
```

IP FINAL/
│
├── **pycache**/                  # Compiled Python files
├── .venv/                        # Virtual environment (ignored in git)
├── Attendance/                   # Attendance CSV files (auto-generated)
│   ├── Attendance-07_31_25.csv
│   ├── Attendance-08_07_25.csv
│   ├── ...
│
├── static/                       # CSS/JS assets (if any)
├── template/                     # HTML templates
│   └── home.html
│
├── app.py                        # Main Flask app
├── haarcascade_frontalface_default.xml  # Haar Cascade model
├── package-lock.json
└── README.md                     # You're here!

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/karthikeya0922/face-recognition-system-.git
cd face-recognition-system-
````

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
```

### 3️⃣ Activate it

* On Windows:

  ```bash
  .venv\Scripts\activate
  ```
* On macOS/Linux:

  ```bash
  source .venv/bin/activate
  ```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

*(If you don’t have a `requirements.txt`, you can create one using `pip freeze > requirements.txt`.)*

---

## ▶️ Run the app

```bash
python app.py
```

Then open your browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧾 Attendance Files

* Each attendance log is automatically saved inside the `Attendance/` folder.
* Files are named with the date (e.g. `Attendance-11_08_25.csv`).

---

## 🧰 Future Improvements

* Add a database (SQLite or Firebase) for attendance tracking.
* Implement user login & role-based dashboards.
* Improve UI with Bootstrap or React frontend.
* Add camera switching and multiple user profiles.

---

## 👨‍💻 Author

**Karthikeya**
AI & ML Enthusiast | Flask Developer | Computer Vision Explorer

---

## 🪪 License

This project is open-source and available under the **MIT License**.

````

---

### 💡 Then commit and push:
```bash
git add README.md
git commit -m "fixed YAML parse issue in README"
git push
````

---
