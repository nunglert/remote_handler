from abc import ABC, abstractmethod


class Task(ABC):
    """This serves as an interface for tasks that should be performed on a 
    remote machine.
    """

    @abstractmethod
    def allocate_data(self):
        """Function handling the allocation of files on the remote file
        system required for the task. 
        
        Example:
            Say you want to do e.g. DFT calculations on another machine. Then 
            this function would handle the logic of setting up the directories
            on the remote machine where the computations are supposed to be 
            performed. Furthermore, it would transfer required input files or 
            create them in-place
        """
        pass

    @abstractmethod
    def setup_slurm(self):
        """Function setting up the slurm logic. This function creates the 
        necessary batch file(s) for slurm to execute the task(s). 

        Example:
            You want to compute DFT for N configs. In `allocate_data` you 
            prepared N directories with the required input files on the remote 
            machine. `setup_slurm` is supposed to create e.g. either a separate
            slurm batch file for each config, running them as N individual 
            slurm jobs or a single slurm batch file that runs them as an array.
        """
        pass
    
    @abstractmethod
    def submit(self):
        """Function submitting the slurm job(s). This should depend on how 
        the slurm logic was handled in `setup_slurm`.
        """
        pass
