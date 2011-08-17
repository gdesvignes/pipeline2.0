import subprocess
import os
import os.path
import time
import re

import queue_managers.generic_interface
import pipeline_utils
import config.basic
import config.email

class MoabManager(queue_managers.generic_interface.PipelineQueueManager):
    def __init__(self, job_basename, property, max_jobs_per_node):
        self.job_basename = job_basename
        self.property = property # the argument to the -q flag in msub
        self.max_jobs_per_node = max_jobs_per_node

    def _exec_check_for_failure(self, cmd):
        """A private method not required by the PipelineQueueManager interface.
            Executes a moab command and checks for moab communication error.

            Input:
                cmd: String command to execute.

            Output:
	        output: Output of the executed command.
                comm_err: Boolean value. True if there was a communication error.
        """

        comm_err_re = re.compile(".*moab may not be running.*")        
 
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE )  

        output, error = pipe.communicate()

        if comm_err_re.search(error):
          print error
          comm_err = True
        else:
          comm_err = False

        return (output, comm_err)

    def submit(self, datafiles, outdir, \
                script=os.path.join(config.basic.pipelinedir, 'bin', 'search.py')):
        """Submits a job to the queue to be processed.
            Returns a unique identifier for the job.

            Inputs:
                datafiles: A list of the datafiles being processed.
                outdir: The directory where results will be copied to.
                script: The script to submit to the queue. (Default:
                        '{config.basic.pipelinedir}/bin/search.py')

            Output:
                jobid: A unique job identifier.
        
            *** NOTE: A pipeline_utils.PipelineError is raised if
                        the queue submission fails.
        """
	
        errorlog = os.path.join(config.basic.qsublog_dir, "'$MOAB_JOBID'.ER") 
        stdoutlog = os.devnull
        #-E needed for $MOAB_JOBID to be defined
        cmd = "msub -E -V -v DATAFILES='%s',OUTDIR='%s' -q %s -l nodes=1:ppn=1,walltime=47:00:00 -N %s -e %s -o %s %s" %\
                   (';'.join(datafiles), outdir, self.property, self.job_basename,\
                      errorlog, stdoutlog, script)
        #pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, \
        #                        stdin=subprocess.PIPE)
        #queue_id = pipe.communicate()[0].strip()
        #pipe.stdin.close()
        queue_id, comm_err = self._exec_check_for_failure(cmd)
        queue_id = queue_id.strip()
        if comm_err:
          errormsg = 'Moab may not be running.'
          raise queue_managers.QueueManagerFatalError(errormsg)
        if not queue_id:
            errormsg  = "No job identifier returned by msub!\n"
            errormsg += "\tCommand executed: %s\n" % cmd
            raise queue_managers.QueueManagerFatalError(errormsg)
        else:
            # There is occasionally a short delay between submission and 
            # the job appearing on the queue, so sleep for 1 second. 
            time.sleep(1)
        return queue_id

    def can_submit(self):
        """Check if we can submit a job
            (i.e. limits imposed in config file aren't met)

            Inputs:
                None

            Output:
                Boolean value. True if submission is allowed.
        """

        running, queued = self.status()
        if ((running + queued) < config.jobpooler.max_jobs_running) and \
            (queued < config.jobpooler.max_jobs_queued):
            return True
        else:
            return False

    def is_running(self, queue_id):
        """Must return True/False whether the job is in the queue or not
            respectively. If there is a moab communication error, assume job 
            is still running.

        Input:
            queue_id: Unique identifier for a job.
        
        Output:
            in_queue: Boolean value. True if the job identified by 'queue_id'
                        is still running.
        """
	state = self._check_job_state(queue_id)
	     
        return ( ('DNE' not in state) and ('Completed' not in state) or ('COMMERR' in state) )

    def _check_job_state(self, queue_id):
        """A private method not required by the PipelineQueueManager interface.
            Return the state of the job in the queue.

            Input:
                queue_id: Unique identifier for a job.

            Output:
	        state: State of the job.
        """

        cmd = "checkjob %s" % queue_id
        #pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, \
        #                        stdin=subprocess.PIPE)
        #status = pipe.communicate()[0]
        #pipe.stdin.close()

        status, comm_err = self._exec_check_for_failure(cmd)
        if comm_err:
          return 'COMMERR'

        lines = status.split('\n')
	for line in lines:
	    if line.startswith("State:"):
	       state = line.split()[1]
               return state
        return 'DNE' # does not exist
	

    def delete(self, queue_id):
        """Remove the job identified by 'queue_id' from the queue.

        Input:
            queue_id: Unique identifier for a job.
        
        Output:
            None
            
            *** NOTE: A pipeline_utils.PipelineError is raised if
                        the job removal fails.
        """
        cmd = "canceljob %s" % queue_id
        pipe = subprocess.Popen(cmd, shell=True)
        
        # Wait a few seconds a see if the job is still being tracked by
        # the queue manager, or if it marked as exiting.
        time.sleep(5)

        state = self._check_job_state(queue_id)
        if ('Completed' not in state) and ('Canceling' not in state) and ('DNE' not in state):
	    errormsg  = "The job (%s) is still in the queue " % queue_id
	    errormsg += "and is marked as state '%s'!\n" % state
            raise pipeline_utils.PipelineError(errormsg)

    def status(self):
        """Return a tuple of number of jobs running and queued for the pipeline

        Inputs:
            None

        Outputs:
            running: The number of pipeline jobs currently marked as running 
                        by the queue manager.
            queued: The number of pipeline jobs currently marked as queued 
                        by the queue manager.
        """
        numrunning = 0
        numqueued = 0
        cmd = "showq -n -w class=%s" % self.property
        #pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, \
        #                        stdin=subprocess.PIPE)
        #jobs = pipe.communicate()[0]
        #pipe.stdin.close()
 
        jobs, comm_err = self._exec_check_for_failure(cmd)
        # what do we want to do with a moab comm err here?

        if comm_err:
          return (9999, 9999)

        lines = jobs.split('\n')
        for line in lines:
            if line.startswith(self.job_basename):
                if 'Running' in line.split()[2]:
                    numrunning += 1
                elif 'Idle' in line.split()[2]:
                    numqueued += 1
        return (numrunning, numqueued)

    def _get_stderr_path(self, jobid_str):
        """A private method not required by the PipelineQueueManager interface.
            Return the path to the error log of the given job, 
            defined by its queue ID.

            Input:
                queue_id: Unique identifier for a job.

            Output:
                stderr_path: Path to the error log file provided by queue 
                        manger for this job.
        
            NOTE: A ValueError is raised if the error log cannot be found.
        """

        stderr_path = os.path.join(config.basic.qsublog_dir, "%s.ER" % jobid_str)
	                              
        if not os.path.exists(stderr_path):
            raise ValueError("Cannot find error log for job (%s): %s" % \
                        (jobid_str, stderr_path))
        return stderr_path

    def had_errors(self, queue_id):
        """Given the unique identifier for a job, return if the job 
            terminated with an error or not.

        Input:
            queue_id: Unique identifier for a job.
        
        Output:
            errors: A boolean value. True if this job terminated with an error.
                    False otherwise.
        """

        try:
            errorlog = self._get_stderr_path(queue_id)
        except ValueError:
            errors = True
        else:
            if os.path.getsize(errorlog) > 0:
                errors = True
            else:
                errors = False
        return errors

    def get_errors(self, queue_id):
        """Return content of error log file for a given queue ID.
        
            Input:
                queue_id: Queue's unique identifier for the job.

            Output:
                errors: The content of the error log for this job (a string).
        """
        try:
            errorlog = self._get_stderr_path(queue_id)
        except ValueError, e:
            errors = str(e)
        else:
            if os.path.exists(errorlog):
                err_f = open(errorlog, 'r')
                errors = err_f.read()
                err_f.close()
        return errors

