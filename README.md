# skience2024

Welcome!

This is the github Project for the 2024 Skience Winter School on Volcano Seismology. Training material for the lectures and practices is provided here.

https://www.skience.de/2024


## Setup of Conda Environment for Skience24

[Miniconda/Anaconda3](https://docs.conda.io/en/latest/miniconda.html) has to be installed for this to work!

1) Download the [__skience24.yaml__](https://raw.githubusercontent.com/heinerigel/skience2024/main/skience24.yaml) file from this repository! This file specifies all required packages for the environment. Warning: make sure the content of the file is correct and not raw html!

2) Using the Anaconda Prompt (or your favorite console if it works): use conda to create an environment: 
  
   ` conda env create -f <path_to_yaml_file> `

3) After this terminated successfully, you should be able to list the environment: 
   
   ` conda env list `
   
   and to activate it using: 
   
   ` conda activate skience24 `

   When activated, your command line should show:
   
   ` (skience24) $ `  
   
4) Test the environment using: 
   
   (skience24) $ ` obspy-runtests --report`
   
   (skience24) $ ` msnoise utils test `
   
5) To eventually delete the environment again type (after the workshop, of course):

    ` conda env remove --name skience24 `

6) Clone the git repository

    in the console, in a folder of your choice (change directory with "cd"), activate the environment and run:
   
    (skience24) $ ` git clone https://github.com/heinerigel/skience2024.git `

If you have issues with the setup, please share the error messages on Mattermost -> Channel "Installation - Software Issues" !

