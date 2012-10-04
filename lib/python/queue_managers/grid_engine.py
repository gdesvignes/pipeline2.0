import subprocess
import os
import os.path
import time
import re

import queue_managers.generic_interface
import pipeline_utils
import config.basic
import config.email
#from xml.etree.ElementTree import ElementTree as ET1
from xml.etree import ElementTree as ET

class GEManager(queue_managers.generic_interface.PipelineQueueManager):
    def __init__(self, job_basename):
        self.job_basename = job_basename
       
        # do a showq to initiate queue list, if comm_err try again
        self.showq_last_update = time.time() - 1
        comm_err = True
        while comm_err:
            self.queue, comm_err = self._showq(update_time=0)

    def _exec_check_for_failure(self, cmd):
        """A private method not required by the PipelineQueueManager interface.
            Executes a Manager command and checks for communication error.

            Input:
                cmd: String command to execute.

            Output:
	        output: Output of the executed command.
                error: Any error messages from the executed command.
                comm_err: Boolean value. True if there was a communication error.
        """

        comm_err_re = re.compile("communication error")        
 
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE )  

        output, error = pipe.communicate()
        if len(error) > 0:
          print error

        if comm_err_re.search(error):
          comm_err = True
        else:
          comm_err = False

        return (output, error, comm_err)

    def submit(self, datafiles, outdir, job_id, resources=[600, 512, 5],\
                script=os.path.join(config.basic.pipelinedir, 'bin', 'search.py'), opts=""):
        """Submits a job to the queue to be processed.
            Returns a unique identifier for the job.

            Inputs:
                datafiles: A list of the datafiles being processed.
                outdir: The directory where results will be copied to.
                job_id: The unique job identifer from the jobtracker database.
                script: The script to submit to the queue. (Default:
                        '{config.basic.pipelinedir}/bin/search.py')

            Output:
                queue_id: A unique queue identifier.
        
            *** NOTE: A pipeline_utils.PipelineError is raised if
                        the queue submission fails.
        """
        
        #filesize = 0 
        #for file in datafiles:
        #    filesize += os.stat(file).st_size   

        #filesize /= 1024.0**3

        #walltime = str( int( self.walltime_per_gb * filesize) ) + ':00:00'
        #print 'Filesize:',filesize,'GB Walltime:', walltime

	try:
	    cputime, memory, fsize = resources
	except:
	    cputime = 10 * 60   # 10 min
	    memory  = 512       # 512 MB
	    fsize   = 5         # 5 GB
	
        errorlog = config.basic.qsublog_dir
        stdoutlog = config.basic.qsublog_dir

	if opts:
	    opts = ",OPTIONS='%s'"%opts

	if config.basic.use_HPSS:
	    hpss_opt = ",hpss=1"
        else: hpss_opt = ""  

	if config.basic.use_sps:
	    sps_opt = ",sps=1"
        else: sps_opt = ""  

	# Submit
        cmd = "qsub  -V -v DATAFILES='%s',OUTDIR='%s'%s -l ct=%d,vmem=%dM,fsize=%dG%s%s -N %s -e %s -o %s %s" %\
                   (';'.join(datafiles), outdir, opts, cputime, memory, fsize, hpss_opt, sps_opt, self.job_basename,\
                      errorlog, stdoutlog, script)
        queue_id, error, comm_err = self._exec_check_for_failure(cmd)
	try:
            queue_id = queue_id.split()[2]
        except:
            pass
        
        comm_err_count = 0
        comm_err_lim = 10

        while comm_err:
          comm_err_count += 1
          if comm_err_count > comm_err_lim:
            errormsg = 'Had more than %d communication errors in a row' % comm_err_lim\
                       + ' while trying to submit.\n'
            raise queue_managers.QueueManagerFatalError(errormsg)

          print 'Communication error during submission: waiting 10s\n'
          time.sleep(10)
          queue_id, comm_err = self._get_submitted_queue_id(job_id)
          
        if not queue_id:
            errormsg  = "No job identifier returned by qsub!\n"
            errormsg += "\tCommand executed: %s\n" % cmd
            errormsg += error
            raise queue_managers.QueueManagerFatalError(errormsg)
        else:
            queue, comm_err = self._showq(update_time=0) # update queue immediately

            # There is occasionally a short delay between submission and 
            # the job appearing on the queue, so sleep for 1 second. 
            time.sleep(1)
        return queue_id

    def _get_submitted_queue_id(self, job_id):

        cmd = 'qstat -xml'
        output, error, comm_err = self._exec_check_for_failure(cmd)
        
        if comm_err:
          return None, comm_err

        else:
          tree = ET.XML(output)
	  job_name = self.job_basename
          queue_id = 0

	  # Loop over all jobs
	  for job in list(tree.getiterator('job_list')):
	    # Search for all Job names 
	    if job.findtext('JB_name') == job_name:
		queue_id = job.findtext('JB_job_number')
          return queue_id, comm_err


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
	     
        return ( ('DNE' not in state) or ('COMMERR' in state) )
        #return ( ('DNE' not in state) and ('Completed' not in state) or ('COMMERR' in state) )


    def _check_job_state(self, queue_id):
        """A private method not required by the PipelineQueueManager interface.
            Return the state of the job in the queue.

            Input:
                queue_id: Unique identifier for a job.

            Output:
	        state: State of the job.
        """

        
        queue, comm_err = self._showq()

	queues = queue['running'] + queue['pending'] + queue['suspended'] + queue['error']

	# 
        for job in queues :
	    if job.findtext('JB_job_number') == str(queue_id):
                return job.attrib['state']

        if comm_err:
          return 'COMMERR'

        print "Job %s does not exist in queue" % queue_id
        return 'DNE'


    def delete(self, queue_id):
        """Remove the job identified by 'queue_id' from the queue.

        Input:
            queue_id: Unique identifier for a job.
        
        Output:
            None
            
            *** NOTE: A pipeline_utils.PipelineError is raised if
                        the job removal fails.
        """
        cmd = "qdel %s" % queue_id
        pipe = subprocess.Popen(cmd, shell=True)
        
        # Wait a few seconds a see if the job is still being tracked by
        # the queue manager, or if it marked as exiting.
        time.sleep(5)

        #force queue update
        queue, comm_err = self._showq(update_time=0)

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

        queue, comm_err = self._showq()

        if comm_err:
          return (9999, 9999)
        #elif error:
        #  raise queue_managers.QueueManagerFatalError(error) 

        numrunning = len(queue['running'])
        numqueued = len(queue['pending']) + len(queue['suspended'])

        #lines = jobs.split('\n')
        #for line in lines:
        #    if line.startswith(self.job_basename):
        #        if 'Running' in line.split()[2]:
        #            numrunning += 1
        #        elif 'Idle' in line.split()[2]:
        #            numqueued += 1

        return (numrunning, numqueued)

    def _get_stderr_path(self, queue_id):
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

        stderr_path = os.path.join(config.basic.qsublog_dir, "%s.e%s" % (self.job_basename, queue_id))
	                              
        if not os.path.exists(stderr_path):
            raise ValueError("Cannot find error log for job (%s): %s" % \
                        (queue_id, stderr_path))
        return stderr_path

    def had_errors(self, queue_id):
        """Given the unique identifier for a job, return if the job 
            terminated with an error or not.

        Input:
            queue_id: Unique queue's identifier for a job.
        
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

        if self._check_job_return_status(queue_id):
	    errors = True

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
         
	errors += "\nReturned exit_status %d"%self._check_job_return_status(queue_id) 

        return errors

    def _check_job_return_status(self, queue_id):
        """Private method to check the status of the job after completion.
        
            Input:
                queue_id: Queue's unique identifier for the job.

            Output:
                ret_code: The exit status reported by the queue manager (if no error, returns zero). 
        """
        cmd = 'qacct -j %s -g glast'%queue_id
        output, error, comm_err = self._exec_check_for_failure(cmd)

	ret_code = 0
        if comm_err:
          return None, comm_err
        else:
	    for line in output.splitlines():
	        if 'exit_status' in line:
		    ret_code = int(line.split()[1])
	return ret_code

    def _showq(self, update_time=10):

        if time.time() >= self.showq_last_update + update_time:
            #print "Updating qstat cache ..."

            cmd = 'qstat -xml'
            output, error, comm_err = self._exec_check_for_failure(cmd)

            queue = {'running': [], 'pending': [], 'suspended': [], 'error': [], 'deleted': []}

            if not comm_err:
                if error:
                  raise queue_managers.QueueManagerFatalError(error) 


                xml = ET.fromstring(output)
		# Loop over all jobs
		for job in list(xml.getiterator('job_list')):
		    # Loop over all infos from a single job
		    for info in list(job.getiterator()):
		        if info.tag == 'JB_name' and info.text.startswith(self.job_basename):
			    queue[job.attrib['state']].append(job)

                self.queue = queue
                self.showq_last_update = time.time()
        else:
            queue = self.queue
            comm_err = False
        
        return queue, comm_err

if __name__ == '__main__':
    a = GEManager("Test")
    #a._get_submitted_queue_id("Test")
    a.status()
    
