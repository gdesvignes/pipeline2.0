"""
pipeline_utils.py

Defines utilities that will be re-used by multiple modules/scripts
in the pipeline package.
"""
import os
import os.path
import sys
import subprocess
import types
import traceback
import optparse
import time
import datetime
import string
import tarfile

import debug
import config.basic
import config.processing

class PipelineError(Exception):
    """A generic exception to be thrown by the pipeline.
    """
    def __init__(self, *args, **kwargs):
        super(PipelineError, self).__init__(*args, **kwargs)
        exctype, excval, exctb = sys.exc_info()
        if (exctype is not None) and (excval is not None) and \
                (exctb is not None):
            self.orig_exc_info = exctype, excval, exctb

    def __str__(self):
        msg = super(PipelineError, self).__str__()
        if 'orig_exc_info' in self.__dict__.keys():
            msg += "\n\n========== Original Traceback ==========\n"
            msg += "".join(traceback.format_exception(*self.orig_exc_info))
            msg += "\n(See PipelineError traceback above)\n"
        if msg.count("\n") > 100:
            msg = string.join(msg.split("\n")[:50],"\n")
        return msg


def get_fns_for_jobid(jobid):
    """Given a job ID number, return a list of that job's data files.

        Input:
            jobid: The ID number from the job-tracker DB to get files for.
        
        Output:
            fns: A list of data files associated with the job ID.
    """
    import jobtracker

    query = "SELECT filename " \
            "FROM files, job_files " \
            "WHERE job_files.file_id=files.id " \
                "AND job_files.job_id=%d" % jobid
    rows = jobtracker.query(query)
    fns = [str(row['filename']) for row in rows]
    return fns


def clean_up(jobid):
    """Deletes raw files for a given job ID.

        Input:
            jobid: The ID corresponding to a row from the job_submits table.
                The files associated to this job will be removed.
        Outputs:
            None
    """
    fns = get_fns_for_jobid(jobid)
    for fn in fns:
        remove_file(fn)

def copy_results_to_HPSS(results_dir):
    """Tar all results and copy them to HPSS

    """
    ar_fn = results_dir+'.tar'
    tf = tarfile.open(ar_fn, "w:")
    tf.add(results_dir)
    tf.close()

    if os.path.exists(ar_fn):
        os.remove(ar_fn)
    

    cmd = "rfcp %s %s"%(ar_fn, os.path.join(config.download.datadir, 'results', os.path.split(ar_fn)[-1]))
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read().strip()

def remove_HPSS_file(fn):
    """Delete a file stored on a HPSS file system

	Input:
	    fn: The name of the file to remove.
	Outputs:
	    None
    """
    cmd = "rfrm %s"%fn
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read().strip()


def remove_file(fn):
    """Delete a file (if it exists) and mark it as deleted in the 
        job-tracker DB.

        Input:
            fn: The name of the file to remove.

        Outputs:
            None
    """
    import jobtracker
    if config.basic.use_HPSS:
        remove_HPSS_file(fn)
    else:
	if os.path.exists(fn):
	    os.remove(fn)
	    print "Deleted: %s" % fn
	jobtracker.query("UPDATE files " \
			 "SET status='deleted', " \
			     "updated_at='%s', " \
			     "details='File was deleted' " \
			 "WHERE filename='%s'" % \
			 (jobtracker.nowstr(), fn))

def get_hpss_file_size(filename):
    """Return the size of a file in HPSS space
    """
    cmd = "rfstat %s"%filename
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
stdin=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read().strip()

    try:
        size = int(pipe.split('\n')[6].split(':')[1])
    except:
        size = -1
    return size


def get_file_size(filename):
    """Return the size of the file using functions depending of the 
    pipeline

    Inputs:
        Filename
    Output:
        The size of the file (in bytes), -1 in case of error
    """
    if config.basic.use_HPSS:
        return get_hpss_file_size(filename)
    else:
        if os.path.exists(filename):
            actualsize = os.path.getsize(filename)
        else:
            actualsize = -1
        return actualsize


def can_add_file_palfa(fn, verbose=False):
    """Checks a file to see if it should be added to the 'files'
        table in the jobtracker DB.

        Input:
            fn: The file to check.
            verbose: Print messages to stdout. (Default: be silent).

        Outputs:
            can_add: Boolean value. True if the file should be added. 
                    False otherwise.
    """
    import jobtracker
    import datafile
    try:
        datafile_type = datafile.get_datafile_type([fn])
    except datafile.DataFileError, e:
        if verbose:
            print "Unrecognized data file type: %s" % fn
        return False
    parsedfn = datafile_type.fnmatch(fn)
    if parsedfn.groupdict().setdefault('beam', '-1') == '7':
        if verbose:
            print "Ignoring beam 7 data: %s" % fn
        return False
    # Check if file is already in the job-tracker DB
    files = jobtracker.query("SELECT * FROM files " \
                             "WHERE filename LIKE '%%%s'" % os.path.split(fn)[-1])
    if len(files):
        if verbose:
            print "File is already being tracked: %s" % fn
        return False
    return True

def can_add_file_generic(fn, verbose=False):
    """Checks a file to see if it should be added to the 'files'
        table in the jobtracker DB.

        Input:
            fn: The file to check.
            verbose: Print messages to stdout. (Default: be silent).

        Outputs:
            can_add: Boolean value. True if the file should be added. 
                    False otherwise.
    """
    import jobtracker

    # Check if file is already in the job-tracker DB
    files = jobtracker.query("SELECT * FROM files " \
                             "WHERE filename LIKE '%%%s'" % os.path.split(fn)[-1])
    if len(files):
        if verbose:
            print "File is already being tracked: %s" % fn
        return False
    return True

def execute(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, dir=None): 
    """Execute the command 'cmd' after logging the command
        to STDOUT. Execute the command in the directory 'dir',
        which defaults to the current directory is not provided.

        Output standard output to 'stdout' and standard
        error to 'stderr'. Both are strings containing filenames.
        If values are None, the out/err streams are not recorded.
        By default stdout is subprocess.PIPE and stderr is sent 
        to sys.stderr.

        Returns (stdoutdata, stderrdata). These will both be None, 
        unless subprocess.PIPE is provided.
    """
    # Log command to stdout
    if debug.SYSCALLS:
        sys.stdout.write("\n'"+cmd+"'\n")
        sys.stdout.flush()

    stdoutfile = False
    stderrfile = False
    if type(stdout) == types.StringType:
        stdout = open(stdout, 'w')
        stdoutfile = True
    if type(stderr) == types.StringType:
        stderr = open(stderr, 'w')
        stderrfile = True
    
    # Run (and time) the command. Check for errors.
    pipe = subprocess.Popen(cmd, shell=True, cwd=dir, \
                            stdout=stdout, stderr=stderr)
    (stdoutdata, stderrdata) = pipe.communicate()
    retcode = pipe.returncode 
    if retcode < 0:
        raise PipelineError("Execution of command (%s) terminated by signal (%s)!" % \
                                (cmd, -retcode))
    elif retcode > 0:
        raise PipelineError("Execution of command (%s) failed with status (%s)!" % \
                                (cmd, retcode))
    else:
        # Exit code is 0, which is "Success". Do nothing.
        pass
    
    # Close file objects, if any
    if stdoutfile:
        stdout.close()
    if stderrfile:
        stderr.close()

    return (stdoutdata, stderrdata)
def get_modtime(file, local=False):
    """Get modification time of a file.

        Inputs:
            file: The file to get modification time for.
            local: Boolean value. If true return modtime with respect 
                to local timezone. Otherwise return modtime with respect
                to GMT. (Default: GMT).

        Outputs:
            modtime: A datetime.datetime object that encodes the modification
                time of 'file'.
    """
    if local:
        modtime = datetime.datetime(*time.localtime(os.path.getmtime(file))[:6])
    else:
        modtime = datetime.datetime(*time.gmtime(os.path.getmtime(file))[:6])
    return modtime


def get_zaplist_tarball(force_download=False):
    """Download zaplist tarball. If the local version has a
        modification time equal to, or later than, the version
        on the FTP server don't download unless 'force_download'
        is True.

        Input:
            force_download: Download zaplist tarball regardless
                of modification times.

        Outputs:
            None
    """
    import config.processing
    import CornellFTP
    cftp = CornellFTP.CornellFTP()
    
    zaptarfile = os.path.join(config.processing.zaplistdir, "zaplists.tar.gz")
    ftpzappath = "/zaplists/zaplists.tar.gz"
    if force_download or (not os.path.exists(zaptarfile)) or \
            (cftp.get_modtime(ftpzappath) > get_modtime(zaptarfile)):
        # Download the file on the FTP
        cftp.download(ftpzappath, config.processing.zaplistdir)
    else:
        # Do nothing
        Pass

    cftp.close()


class PipelineOptions(optparse.OptionParser):
    def __init__(self, *args, **kwargs):
        optparse.OptionParser.__init__(self, *args, **kwargs)
       
    def parse_args(self, *args, **kwargs):
        # Add debug group just before parsing so it is the last set of
        # options displayed in help text
        self.add_debug_group()
        return optparse.OptionParser.parse_args(self, *args, **kwargs)

    def add_debug_group(self):
        group = optparse.OptionGroup(self, "Debug Options", \
                    "The following options turn on various debugging " \
                    "features in the pipeline. Multiple debugging " \
                    "options can be provided.")
        group.add_option('-d', '--debug', action='callback', \
                          callback=self.debugall_callback, \
                          help="Turn on all debugging modes. (Same as --debug-all).")
        group.add_option('--debug-all', action='callback', \
                          callback=self.debugall_callback, \
                          help="Turn on all debugging modes. (Same as -d/--debug).")
        for m, desc in debug.modes:
            group.add_option('--debug-%s' % m.lower(), action='callback', \
                              callback=self.debug_callback, \
                              callback_args=(m,), \
                              help=desc)
        self.add_option_group(group)

    def debug_callback(self, option, opt_str, value, parser, mode):
        debug.set_mode_on(mode)

    def debugall_callback(self, option, opt_str, value, parser):
        debug.set_allmodes_on()
