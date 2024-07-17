from __future__ import print_function
from optparse import OptionParser
import sys
import subprocess
import tempfile
import time


def main():
    """
    Launches a job with specified parameters and optionally monitors its status and cleans up afterward.

    Command-line Options
    --------------------
    -g, --go               Don't wait for the job to finish [Default: False]
    -o, --out_file         Output file path for job logs
    -e, --err_file         Error file path for job logs
    -J, --job_name         Name of the job
    -q, --queue            Job queue [Default: general]
    -n, --cpu              Number of CPUs for the job [Default: 1]
    -m, --mem              Memory allocation for the job (in MB)
    -t, --time             Maximum execution time for the job

    Returns
    ---------
    None
    """
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option(
        "-g",
        dest="go",
        default=False,
        action="store_true",
        help="Don't wait for the job to finish [Default: %default]",
    )

    parser.add_option("-o", dest="out_file")
    parser.add_option("-e", dest="err_file")

    parser.add_option("-J", dest="job_name")

    parser.add_option("-q", dest="queue", default="general")
    parser.add_option("-n", dest="cpu", default=1, type="int")
    parser.add_option("-m", dest="mem", default=None, type="int")
    parser.add_option("-t", dest="time", default=None)

    (options, args) = parser.parse_args()

    cmd = args[0]

    main_job = Job(
        cmd,
        name=options.job_name,
        out_file=options.out_file,
        err_file=options.err_file,
        queue=options.queue,
        cpu=options.cpu,
        mem=options.mem,
        time=options.time,
    )
    main_job.launch()

    if options.go:
        time.sleep(1)

        # find the job
        if not main_job.update_status:
            time.sleep(1)

        # delete sbatch
        main_job.clean()

    else:
        time.sleep(10)

        # find the job
        if not main_job.update_status():
            time.sleep(10)

        # wait for it to complete
        while main_job.update_status() and main_job.status in ["PENDING", "RUNNING"]:
            time.sleep(30)

        print("%s %s" % (main_job.name, main_job.status), file=sys.stderr)

        # delete sbatch
        main_job.clean()


def multi_run(jobs, max_proc=None, verbose=False, launch_sleep=2, update_sleep=20):
    """
    Launch multiple jobs sequentially and monitor their statuses until all jobs have finished.

    Parameters
    ------------
    jobs : list
        List of Job objects representing the jobs to be executed.
    max_proc : int, optional
        Maximum number of jobs to run concurrently. Default is None (runs all jobs concurrently).
    verbose : bool, optional
        If True, print job names and commands to stderr as they are launched and completed. Default is False.
    launch_sleep : int, optional
        Time to sleep (in seconds) after launching each job. Default is 2 seconds.
    update_sleep : int, optional
        Time to sleep (in seconds) between updating job statuses. Default is 20 seconds.

    Returns
    ---------
    None
    """
    total = len(jobs)
    finished = 0
    running = 0
    active_jobs = []
    
    if max_proc is None:
        max_proc = len(jobs)
    
    while finished + running < total:
        # launch jobs up to the max
        while running < max_proc and finished + running < total:
            # launch
            jobs[finished + running].launch()
            time.sleep(launch_sleep)
            if verbose:
                print(
                    jobs[finished + running].name,
                    jobs[finished + running].cmd,
                    file=sys.stderr,
                )

            # save it
            active_jobs.append(jobs[finished + running])
            running += 1

        # sleep
        time.sleep(update_sleep)
        
        # update all statuses
        multi_update_status(active_jobs)
        
        # update active jobs
        active_jobs_new = []
        for i in range(len(active_jobs)):
            if active_jobs[i].status in ["PENDING", "RUNNING"]:
                active_jobs_new.append(active_jobs[i])
            else:
                if verbose:
                    print(
                        "%s %s" % (active_jobs[i].name, active_jobs[i].status),
                        file=sys.stderr,
                    )

                running -= 1
                finished += 1

        active_jobs = active_jobs_new


def multi_update_status(jobs, max_attempts=3, sleep_attempt=5):
    """
    Update the status of multiple Job objects by querying job status using `sacct` command.

    Parameters
    ------------
    jobs : list
        List of Job objects whose statuses need to be updated.
    max_attempts : int, optional
        Maximum number of attempts to query job status. Default is 3.
    sleep_attempt : int, optional
        Time to sleep (in seconds) between attempts to query job status. Default is 5 seconds.

    Returns
    ---------
    None
    """
    
    for j in jobs:
        j.status = None
    
    # try multiple times because sometimes it fails
    attempt = 0
    while attempt < max_attempts and [j for j in jobs if j.status is None]:

        if attempt > 0:
            time.sleep(sleep_attempt)

        sacct_str = subprocess.check_output("sacct", shell=True)
        sacct_str = sacct_str.decode("UTF-8")
        
        # split into job lines
        sacct_lines = sacct_str.split("\n")
        for line in sacct_lines[2:]:
            a = line.split()
            
            if a != []:
                try:
                    line_id = int(a[0].split(".")[0])
                except Exception as ex:
                    print(ex)
                    line_id = None

                # check call jobs for a match
                for j in jobs:
                    if line_id == j.id:
                        j.status = a[5].split(".")[0]
        
        attempt += 1
    
class Job:
    """
    Represents a job to be submitted to SLURM.

    Attributes
    ----------
    cmd : str
        Command to be executed.
    name : str
        Name of the job.
    out_file : str, optional
        Output file path for job logs.
    err_file : str, optional
        Error file path for job logs.
    sb_file : str, optional
        Path to the SLURM batch script file.
    queue : str, optional
        Queue name for job submission. Default is 'standard'.
    cpu : int, optional
        Number of CPUs required for the job. Default is 1.
    mem : int, optional
        Memory allocation (in MB) required for the job.
    time : str, optional
        Maximum execution time for the job (format: days-hours:minutes:seconds).
    gpu : int, optional
        Number of GPUs required for the job. Default is 0.
    gres : str, optional
        Generic resource specification for GPUs. Default is 'gpu:p100'.
    constraint : str, optional
        Specific node constraint for the job.

    Methods
    -------
    flash():
        Determine if the job can run on the flash queue based on its time requirements.
    launch():
        Create and launch the SLURM batch script, and save the job ID.
    update_status(max_attempts=3, sleep_attempt=5):
        Update the job status using 'sacct' command and return True if successful, False otherwise.
    """
    
    def __init__(
        self,
        cmd,
        name,
        out_file=None,
        err_file=None,
        sb_file=None,
        # account=None,
        queue="standard",
        cpu=1,
        mem=None,
        time=None,
        gpu=0,
        gres="gpu:p100",
        constraint=None,
    ):
        self.cmd = cmd
        self.name = name
        self.out_file = out_file
        self.err_file = err_file
        self.sb_file = sb_file
        self.queue = queue
        self.cpu = cpu
        self.mem = mem
        self.time = time
        self.gpu = gpu
        self.gres = gres
        self.constraint = constraint
        self.id = None
        self.status = None

    def flash(self):
        """Determine if the job can run on the flash queue by parsing the time."""

        day_split = self.time.split("-")
        if len(day_split) == 2:
            days, hms = day_split
        else:
            days = 0
            hms = day_split[0]

        hms_split = hms.split(":")
        if len(hms_split) == 3:
            hours, mins, secs = hms_split
        elif len(hms_split) == 2:
            hours = 0
            mins, secs = hms_split
        else:
            print("Cannot parse time: ", self.time, file=sys.stderr)
            exit(1)

        hours_sum = 24 * int(days) + int(hours) + float(mins) / 60

        return hours_sum <= 4

    def launch(self):
        """Create and launch the SLURM batch script, and save the job ID."""

        # make sbatch script
        if self.sb_file is None:
            sbatch_tempf = tempfile.NamedTemporaryFile()
            sbatch_file = sbatch_tempf.name
        else:
            sbatch_file = self.sb_file
        sbatch_out = open(sbatch_file, "w")

        print("#!/bin/bash\n", file=sbatch_out)
        if self.gpu > 0:
            gres_str = "--gres=%s" % self.gres
            print("#SBATCH -p %s" % self.queue, file=sbatch_out)
            print("#SBATCH %s \n" % (gres_str), file=sbatch_out)
        else:
            print("#SBATCH -p %s" % self.queue, file=sbatch_out)
        print("#SBATCH -n 1", file=sbatch_out)
        print("#SBATCH -c %d" % self.cpu, file=sbatch_out)
        if self.name:
            print("#SBATCH -J %s" % self.name, file=sbatch_out)
        if self.out_file:
            print("#SBATCH -o %s" % self.out_file, file=sbatch_out)
        if self.err_file:
            print("#SBATCH -e %s" % self.err_file, file=sbatch_out)
        if self.mem:
            print("#SBATCH --mem %d" % self.mem, file=sbatch_out)
        if self.time:
            print("#SBATCH --time %s" % self.time, file=sbatch_out)
        if self.constraint:
            print("#SBATCH --constraint %s" % self.constraint, file=sbatch_out)
        print(self.cmd, file=sbatch_out)

        sbatch_out.close()

        # launch it; check_output to get the id
        launch_str = subprocess.check_output("sbatch %s" % sbatch_file, shell=True)

        # e.g. "Submitted batch job 13861989"
        self.id = int(launch_str.split()[3])

    def update_status(self, max_attempts=3, sleep_attempt=5):
        """
        Update the job's status using 'sacct' command.

        Parameters
        ------------
        max_attempts : int, optional
            Maximum number of attempts to query job status. Default is 3.
        sleep_attempt : int, optional
            Time to sleep (in seconds) between attempts to query job status. Default is 5 seconds.

        Returns
        ---------
        bool
            True if job status is updated successfully, False otherwise.
        """

        status = None

        attempt = 0
        while (attempt < max_attempts) and (status is None):
            if attempt > 0:
                time.sleep(sleep_attempt)

            sacct_str = subprocess.check_output("sacct", shell=True)
            sacct_str = sacct_str.decode("UTF-8")

            sacct_lines = sacct_str.split("\n")
            for line in sacct_lines[2:]:
                a = line.split()
                
                if a != []:
                    try:
                        line_id = int(a[0])
                    except Exception as ex:
                        print(ex)
                        line_id = None

                    if line_id == self.id:
                        status = a[5]

            attempt += 1

        if status is None:
            return False
        else:
            self.status = status
            return True

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
