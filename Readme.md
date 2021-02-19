A repository for code related to Google HashCode challenges.

See https://hashcode.withgoogle.com/past_editions.html.

Each problem is in a separate folder organised by year.

This project is using python 3 with a virtual environment.
Your can generate it with the command line or in your IDE (like PyCharm).

```
virtualenv venv -p {path to a python 3 installation}
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Example after installing python3.9:
```
which pip3.9 # for me all python 3.9 programs are here : /Library/Frameworks/Python.framework/Versions/3.9/bin/
pip3.9 install --upgrade pip
pip3.9 install virtualenv

# run the virtial env you want, and create venv from the python version you want
/Library/Frameworks/Python.framework/Versions/3.9/bin/virtualenv -p /Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9 venv
source venv/bin/activate
which pip # should be the one in your folder/venv
which python # should be local one
python --version # should be 3.9
pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook # launch jupyter
```

If you want to reach out, feel free to reach me on LinkedIn: https://www.linkedin.com/in/emmanuel-charon-03725357/ .