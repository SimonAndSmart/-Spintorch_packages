# Spintorch_packages
necessary packages to run the main code

The package is mainly done by a-papp you can find it at: https://github.com/a-papp/SpinTorch

in the folder 'example_code' you can see the following:
- an interactive local version Jupiter notebook code.
- a 'multi_train_test_on_HPC' folder allowing you to train, test, and analyse the model under different combinations of parameters.
  within the 'multi_train_test_on_HPC' folder you will see:
  - a 'Creat_json' Jupiter notebook which allows you to batch setting all the parameters into JSON files.
  - a PBS file which submits multi jobs to the HPC (you will need to change the 4th line '#PBS -J 1-60' if you have different numbers of JSON files.)
  - a script file which helps you git pull the package and submit the PBS file.
  - a 'read_acc_result' Jupiter notebook which reads the 'plot' folder inside the 'Spintorch_packages' on HPC when the job has finished.
  - and finally, a Python file which is the main code will be executed by the PBS files.

  Note: when you want to run this example, executed the 'Creat_json' locally and then transfer the entire 'multi_train_test_on_HPC' folder to the HPC.
        Then, cd to the folder path in HPC and use 'chmod +x qsub_with_git_pull.sh' to give the script permission to execute.
        After this start the HPC jobs with './qsub_with_git_pull.sh' and you can wait until it finishes.
        When all or part of the jobs have finished, download the 'plot' file inside the 'Spintorch_packages' to your local PC, copy its path and replace the
        'path' in the 'read_acc_result' Jupiter notebook and you can start the analysis.
