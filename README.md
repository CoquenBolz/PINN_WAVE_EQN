# PINN_WAVE_EQN

This follows greatly from https://github.com/okada39/pinn_wave/blob/master/README.md with the exceptions that the classes network and pinn are consolidated to be functions in main.py

# Version

To run this script you will need to install a certain version of python (vers 3.13.9). 
If you must, just use within version 3.5 - 3.13 so that you can install tensorflow

To install the correct version
* Go to https://www.python.org/downloads/
* Scroll down to versions
* Click 3.13.9
* Scroll down to versions
* Click either windows installer 64 bit or macOS installer 64 bit depending on your device
* Run the installer once it downloads
* Click Install Now
* You should have Python 3.13 downloaded to your OS now!

To install Visual Studio Code
* Click here https://code.visualstudio.com/download
* Click the Big Blue Box under the Image that Best represents what kind of device you're using
* Follow instructions from installer

To Open my here repository in Visual Studios Code
* Click Files in the top right corner
* Click Open Folder
* Find where you downloaded this repository to (could be your downloads folder or wherever) and click it to open this folder

Install the Python Extension
* On the side bar you should find an icon such that if you hover over it it says "Extensions"
  * It also suffices to press ctrl + shift + x
* There should be a search bar at the top of the side panel of your screen where you may type "Python"
* Try to find the Extension Published by Microsoft simply titled "Python" and click install
* You should now have the python extension downloaded for use

Setup this repository in vscode to run
* Select Python Interpreter
  * At the top of the window is a Search bar (perhaps with the repository name typed in it), click on it
  * You should now see a dropdown menu from the seach bar, Click on "Show and Run Commands >" (should be listed second)
  * *Alternatively it would also be an option to use the shortcut ctrl + shift + P*
  * You should now see another new dropdown menu, Search "Python: Select Interpreter"
  * Select Python 3.13.9 --Yayy you did it
* Open CMD Terminal
  * At the top of the window you may find a button titled "Terminal" click on it
    * if you do not see terminal perhaps try clicking on a button at the top labeled with "..." to reveal it
  * select "New Terminal"
  * Now you should see a new terminal opened at the bottom of your screen (ignore any errors if they are there)
  * At the top of this new terminal, towards the right should be a "+ ˅" button, click on the "˅"
  * In the dropdown menu select "Command Prompt"
  * You should now have a  Command Prompt terminal which you should be able to select by clicking "cmd" on a panel to the right side of your terminal
* Set up Virtual environment
  * in the Command Prompt Terminal (cmd) you just opened type `python -m venv [ENVIRONMENT_NAME]` (you can type whatever you desire in [ENVIRONMENT_NAME], it will simply be the name of the virtual environment you use to run our code)
  * there should be a notification asking to update your interpreter (or something of the sort) to include your new virtual environment, click yes
  * now run `\[ENVIRONMENT_NAME]\Scripts\Activate` in the cmd, to activate your environment
* Download necessary Packages/Libraries
  * for all the packages or libraries typed after the word "import" at the top of main, run the command `python pip install [PACKAGE_NAME]`
  * Here is the list of Packages:
    * numpy
    * tensorflow
    * matplotlib
    * scipy
