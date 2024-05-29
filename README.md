***
# “Yes” or “No” using Head Pose Estimation to Answer thought-provoking technological Questions
This project makes use of Head pose estimation to determine if the user is nodding or shaking their head. The user is posed with a question, then prompted to answer with a "yes" or "no" answer and then can choose on either displaying the answer if satisfied or refreshing and retrying if needed. It also allows the user to see if their answer is being recorded correctly or prompts them to retry if the head movement is ambiguous. The user can then go through all questions, answer them accordingly and after quitting the video screen, see a compiled list of question, answers and the timestamp at which they answered the questions.
***
## Installation

To use this project, you will need to have Python 3.x installed on your local machine. You can download and install the latest version of Python from the official Python website.

Once you have Python installed, you can clone this repository using Git or download the ZIP file and extract it to your local machine.

Next, navigate to the project directory and install the required libraries and dependencies by running the following command in your terminal:

```bash
  pip install -r requirements.txt
```
***
## Usage/Examples

To use this project, run the following command in your terminal:

```bash
  python Head_movement_estimation.py
```

This will launch the project and open a new window showing the webcam output. To answer the questions simply nod for "yes" or shake your head for "no".

Press 's' key on your keyboard to display your answer.

Press 'r' key to reset your answer and retry.

To exit the project, simply press the 'q' key on your keyboard.
***
## Files and Directories

This repository contains the following files and directories:

|  File  | Description |
| ------ | ----------- |
| 'Head_movement_estimation.py' | the main Python script that runs the project |
| 'README.md' | this file contains instructions for installing and using the project |
| 'requirements.txt' | a file that lists the required libraries and dependencies for running the project |
***
