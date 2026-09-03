
#  BURD

BURD is a mobile application that uses deep learning to identify bird species from photographs. Users can provide an image of a bird, and the application sends it to a backend inference server where a trained machine learning model analyzes the image and returns a prediction.

The project combines **mobile development**, **backend development**, and **machine learning** to provide a simple bird species recognition experience.

---

##  Features

-  Identify bird species from images
-  AI-powered image classification
-  Cross-platform mobile application built with Flutter
-  Backend inference server for running the machine learning model
-  Simple and user-friendly interface

---


The Flutter application handles the user interface and image selection, while the backend server processes the image and performs machine learning inference.

---

##  Tech Stack

### Mobile Application

* Flutter
* Dart

### Backend & Machine Learning

The machine learning inference is handled by a separate backend server.

 **Backend Inference Server:**
[https://github.com/Sadotib/burd_server](https://github.com/Sadotib/burd_server)

---

##  Getting Started

### Prerequisites

Make sure you have Flutter installed on your system.

You can verify your Flutter installation with:

```bash
flutter doctor
```

---

### Clone the Repository

```bash
git clone https://github.com/Sadotib/burd.git
cd burd
```

---

### Install Dependencies

```bash
flutter pub get
```

---

### Run the Application

```bash
flutter run
```

Make sure you have an Android emulator, iOS simulator, or physical device connected.

---

##  Backend Setup

The BURD application requires the backend inference server to process bird images.

Clone the backend repository:

```bash
git clone https://github.com/Sadotib/burd_server.git
cd burd_server
```

For backend installation and setup instructions, visit:

🔗 **BURD Backend Inference Server:**
[https://github.com/Sadotib/burd_server](https://github.com/Sadotib/burd_server)

Once the backend server is running, configure the BURD mobile application to communicate with the server.

---

##  How It Works

1.  The user selects or captures an image of a bird.
2.  The BURD mobile application sends the image to the backend server.
3.  The backend preprocesses the image.
4.  The trained machine learning model performs inference.
5.  The predicted bird species is returned to the mobile application.
6.  The result is displayed to the user.

---

## 📂 Project Structure

```text
burd/
│
├── android/          # Android-specific configuration
├── ios/              # iOS-specific configuration
├── lib/              # Flutter application source code
├── assets/           # Application assets
├── test/             # Application tests
├── web/              # Web platform configuration
├── windows/          # Windows platform configuration
├── linux/            # Linux platform configuration
├── macos/            # macOS platform configuration
│
├── pubspec.yaml      # Flutter dependencies and configuration
└── README.md
```

---


---

##  Project Goal

The goal of BURD is to explore the integration of **machine learning** and **mobile development** by bringing a bird image classification model into a practical application.

This project demonstrates the workflow of:

* Training and using a machine learning model
* Preparing the model for inference
* Deploying inference through a backend server
* Connecting a Flutter application to the backend
* Sending images from a mobile application
* Receiving and displaying AI predictions

---

