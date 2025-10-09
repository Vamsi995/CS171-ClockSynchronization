# CS171 Clock Synchronization Autograder

## 🕒 About

This repository contains the **autograder script** for the **CS171: Clock Synchronization** assignment.  
It is designed to automatically test the implemenation of programming assignment 1 - clock synchronization.


---

## ⚙️ Setup

### Requirements
Make sure you have the following installed:
- **Python 3.6+**
- Ensure that you are using a virtual environment and not installing the required libraries to your base system python.
- Required Python packages (install with pip):
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Usage

Run the Autograder

```bash
python autograder.py --submissions <submission folder containing the makefile>
```

For example:

Consider the below directory structure where all your files are under a directory named project_dir. 

```
projet_dir
├── client.py          # client script
├── server.py          # server script
├── makefile           # makefile
└── network.py         # network script
```

Then your command for the autograder will be as follows:

```bash
python autograder.py --submissions project_dir
```

Pass the directory in which the makefile is present to the ```---submissions``` command line argument.



For debugging your code, the logs from the autograder script could be seen in the ```autograder.log``` file which automatically gets created once you run the autograder script.
