import os.path
import sys
import os
import shutil
import time
import re
import threading
import traceback

import debug
import mailer
import OutStream
import datafile
import database
import jobtracker
import pipeline_utils
import config.background
import config.download
import config.email
import config.basic

import DownloaderSPAN512 

dlm_cout = OutStream.OutStream("Download Module", \
                        os.path.join(config.basic.log_dir, "downloader.log"), \
                        config.background.screen_output)


def check_download_attempts():
    """For each download attempt with status 'downloading' check
        to see that its thread is still active. If not, mark it
        as 'unknown', and mark the file as 'unverified'.
    """
    attempts = jobtracker.query("SELECT * FROM download_attempts " \
                                "WHERE status='downloading'")

    active_ids = [int(t.getName()) for t in threading.enumerate() \
                            if isinstance(t, DownloadThread)]

    for attempt in attempts:
        if attempt['id'] not in active_ids:
            dlm_cout.outs("Download attempt (ID: %d) is no longer running." % \
                            attempt['id'])
            queries = []
            queries.append("UPDATE files " \
                           "SET status='unverified', " \
                                "updated_at='%s', " \
                                "details='Download thread is no longer running' "
                           "WHERE id=%d" % (jobtracker.nowstr(), attempt['file_id']))
            queries.append("UPDATE download_attempts " \
                           "SET status='unknown', " \
                                "updated_at='%s', " \
                                "details='Download thread is no longer running' "
                           "WHERE id=%d" % (jobtracker.nowstr(), attempt['id']))
            jobtracker.query(queries)


def can_request_more():
    """Returns whether Downloader can request more restores.
        This is based on took disk space allowed for downloaded
        file, disk space available on the file system, and maximum
        number of active requests allowed.

    Inputs:
        None
    Output:
        can_request: A boolean value. True if Downloader can make a request.
                        False otherwise.
    """
    # Note: Files are restored in pairs (so we multiply by 2)
    active_requests = jobtracker.query("SELECT IFNULL(SUM(numrequested), 0) " \
                                       "FROM requests " \
                                       "WHERE status='waiting'", fetchone=True)
    to_download = jobtracker.query("SELECT * FROM files " \
                                   "WHERE status NOT IN ('downloaded', " \
                                                        "'added', " \
                                                        "'deleted', " \
                                                        "'terminal_failure')")
    if active_requests == None:
	active_requests = 0
    num_to_restore = active_requests
    num_to_download = len(to_download)
    used = get_space_used()
    reserved = get_space_committed()

    can_request = ((num_to_restore+num_to_download) < config.download.numrestored) and \
            (used+reserved < config.download.space_to_use)
    return can_request




def get_space_used():
    """Return space used by the download directory (config.download.datadir)

    Inputs:
        None
    Output:
        used: Size of download directory (in bytes)
    """
    files = jobtracker.query("SELECT * FROM files " \
                             "WHERE status IN ('added', 'downloaded', 'unverified')")

    total_size = 0
    for file in files:
        total_size += int(file['size'])
    return total_size

def get_space_available():
    """Return space available

    Inputs:
        None
    Output:
        available: Size of download directory (in bytes)
    """
    return config.download.space_to_use - get_space_used()

def get_space_committed():
    """Return space reserved to files to be downloaded.

        Inputs:
            None
        Outputs:
            reserved: Number of bytes reserved by files to be downloaded.
    """
    reserved = jobtracker.query("SELECT SUM(size) FROM files " \
                                "WHERE status IN ('downloading', 'new', " \
                                                 "'retrying', 'failed')", \
                                fetchone=True)
    if reserved is None:
        reserved = 0
    return reserved


def run():
    """Perform a single iteration of the downloader's loop.

        Inputs:
            None
        Outputs:
            numsuccess: The number of successfully downloaded files 
                        this iteration.
    """
    check_active_requests()
    start_downloads()
    check_download_attempts()
    numsuccess = verify_files()
    recover_failed_downloads()
    check_downloading_requests()
    acknowledge_downloaded_files()
    if can_request_more():
        make_request()
    return numsuccess


def make_request(dbname='default'):
    """Make a request for data to be restored by connecting to the
        data server.
    """
    num_beams = get_num_to_request()
    if not num_beams:
        # Request size is 0
        return
    dlm_cout.outs("Requesting data\nIssuing a request of size %d" % num_beams)

    # Ask to restore num_beams
    db = database.Database(dbname)
    QUERY = "SELECT f.obs_id FROM full_processing as f LEFT JOIN  processing AS p ON f.obs_id = p.obs_id WHERE f.status='available' AND p.details is NULL LIMIT %d"%num_beams
    db.cursor.execute(QUERY)
    obs_ids = [row[0] for row in db.cursor.fetchall()]

    # Ask for an uuid
    QUERY = "SELECT  UUID();"
    db.cursor.execute(QUERY)
    guid = db.cursor.fetchone()[0]

    if not obs_ids:
        print "There are no files to be restored."
        return

    # Mark the beams for restorations
    for obs_id in obs_ids:
        QUERY = "UPDATE full_processing SET status='requested', guid='%s', updated_at=NOW() WHERE obs_id=%s"%(guid, obs_id)
        db.cursor.execute(QUERY)
    db.conn.close()

    #if guid == "fail":
    #   raise pipeline_utils.PipelineError("Request for restore returned 'fail'.")

    requests = jobtracker.query("SELECT * FROM requests WHERE guid='%s'" % guid)

    if requests:
        # Entries in the requests table exist with this GUID!?
        raise pipeline_utils.PipelineError("There are %d requests in the " \
                               "job-tracker DB with this GUID %s" % \
                               (len(requests), guid))

    jobtracker.query("INSERT INTO requests ( " \
                        "numbits, " \
                        "numrequested, " \
                        "file_type, " \
                        "guid, " \
                        "created_at, " \
                        "updated_at, " \
                        "status, " \
                        "details) " \
                     "VALUES (%d, %d, '%s', '%s', '%s', '%s', '%s', '%s')" % \
                     (config.download.request_numbits, num_beams, \
                        config.download.request_datatype, guid, \
                        jobtracker.nowstr(), jobtracker.nowstr(), 'waiting', \
                        'Newly created request'))


def check_active_requests():
    """Check for any requests with status='waiting'. If there are
        some, check if the files are ready for download.
    """

    active_requests = jobtracker.query("SELECT * FROM requests " \
                                       "WHERE status='waiting'")
    for request in active_requests:

	# Check requested status 
	if DownloaderSPAN512.check_request_done(request):
	    dlm_cout.outs("Restore (GUID: %s) has succeeded. Will create file entries.\n" % request['guid'])
	    create_file_entries(request)

	else:
#	    dlm_cout.outs("Request (GUID: %s) has failed.\n" \
#	             "\tDatabase failed to report the data as restored." % request['guid'])
#	    jobtracker.query("UPDATE requests SET status='failed', " \
#                     "details='Request failed. Why ?', " \
#                     "updated_at='%s' " \
#                     "WHERE guid='%s'" % (jobtracker.nowstr(), request['guid']))

            query = "SELECT (TO_SECONDS('%s')-TO_SECONDS(created_at)) " \
                        "AS deltaT_seconds " \
                    "FROM requests " \
                    "WHERE guid='%s'" % \
                        (jobtracker.nowstr(), request['guid'])
            row = jobtracker.query(query, fetchone=True)
            #if row['deltaT_seconds']/3600. > config.download.request_timeout:
            if row/3600. > config.download.request_timeout:
                dlm_cout.outs("Restore (GUID: %s) is over %d hr old " \
                                "and still not ready. Marking " \
                                "it as failed." % \
                        (request['guid'], config.download.request_timeout))
                jobtracker.query("UPDATE requests " \
                                 "SET status='failed', " \
                                    "details='Request took too long (> %d hr)', " \
                                    "updated_at='%s' " \
                                 "WHERE guid='%s'" % \
                    (config.download.request_timeout, jobtracker.nowstr(), \
                            request['guid']))



def create_file_entries(request):
    """Given a row from the requests table in the job-tracker DB
        check the data server for its files and create entries in
        the files table.

        Input:
            request: A row from the requests table.
        Outputs:
            None
    """

    files = DownloaderSPAN512.get_files_infos(request)
    
    total_size = 0
    num_files = 0
    queries = []
    for fn, size in files:
        # Check if it's ok to add the file in the DB (check if it's already in the DB, may add further criteria...)
        if not pipeline_utils.can_add_file_generic(fn): 
            dlm_cout.outs("Skipping %s" % fn)
            continue

        # Insert entry into DB's files table
        queries.append("INSERT INTO files ( " \
                            "request_id, " \
                            "remote_filename, " \
                            "filename, " \
                            "status, " \
                            "created_at, " \
                            "updated_at, " \
                            "size) " \
                       "VALUES ('%s', '%s', '%s', '%s', '%s', '%s', %d)" % \
                       (request['id'], fn, os.path.join(config.download.datadir, os.path.split(fn)[-1]), \
                        'new', jobtracker.nowstr(), jobtracker.nowstr(), size))
        total_size += size
        num_files += 1

    if num_files:
        dlm_cout.outs("Request (GUID: %s) has succeeded.\n" \
                        "\tNumber of files to be downloaded: %d" % \
                        (request['guid'], num_files))
        queries.append("UPDATE requests " \
                       "SET size=%d, " \
                            "updated_at='%s', " \
                            "status='downloading', " \
                            "details='Request has been filled' " \
                       "WHERE id=%d" % \
                       (total_size, jobtracker.nowstr(), request['id']))
    else:
        dlm_cout.outs("Request (GUID: %s) has failed.\n" \
                        "\tThere are no files to be downloaded." % \
                        request['guid'])

        # delete restore since there may be skipped files
	DownloaderSPAN512.delete_stagged_file(request)

	# redefine 'queries' because there are no files to update
	queries = ["UPDATE requests " \
		   "SET updated_at='%s', " \
			"status='failed', " \
			"details='No files to download.' " \
		   "WHERE id=%d" % \
		   (jobtracker.nowstr(), request['id'])]

    jobtracker.query(queries)


def start_downloads():
    """Check for entries in the files table with status 'retrying'
        or 'new' and start the downloads.
    """
    todownload  = jobtracker.query("SELECT * FROM files " \
                                   "WHERE status='retrying' " \
                                   "ORDER BY created_at ASC")
    todownload += jobtracker.query("SELECT * FROM files " \
                                   "WHERE status='new' " \
                                   "ORDER BY created_at ASC")

    for file in todownload:
        if can_download():
            dlm_cout.outs("Initiating download of %s" % \
                            os.path.split(file['filename'])[-1])

            # Update file status and insert entry into download_attempts
            queries = []
            queries.append("UPDATE files " \
                           "SET status='downloading', " \
                                "details='Initiated download', " \
                                "updated_at='%s' " \
                            "WHERE id=%d" % \
                            (jobtracker.nowstr(), file['id']))
            jobtracker.query(queries)
            queries = []
            queries.append("INSERT INTO download_attempts (" \
                                "status, " \
                                "details, " \
                                "updated_at, " \
                                "created_at, " \
                                "file_id) " \
                           "VALUES ('%s', '%s', '%s', '%s', %d)" % \
                           ('downloading', 'Initiated download', jobtracker.nowstr(), \
                                jobtracker.nowstr(), file['id']))
            insert_id = jobtracker.query(queries, fetchone=True)
            attempt = jobtracker.query("SELECT * FROM download_attempts " \
                                       "WHERE id=%d" % insert_id, fetchone=True)
    
            # download(attempt)
            DownloadThread(attempt).start()
        else:
            break


def get_num_to_request():
    """Return the number of files to request given the average
        time to download a file (including waiting time) and
        the amount of space available.

        Inputs:
            None

        Outputs:
            num_to_request: The size of the request.
    """
    #ALLOWABLE_REQUEST_SIZES = [1, 3, 5, 8, 12]
    ALLOWABLE_REQUEST_SIZES = [1, 3, 5, 8]
    avgrate = jobtracker.query("SELECT AVG(files.size/" \
                                "(TO_SECONDS(download_attempts.updated_at)*1/86400. - " \
                                "TO_SECONDS(download_attempts.created_at)*1/86400.)) " \
                               "FROM files, download_attempts " \
                               "WHERE files.id=download_attempts.file_id " \
                                    "AND download_attempts.status='downloaded'", \
                               fetchone=True)
    avgsize = jobtracker.query("SELECT AVG(size/numrequested) FROM requests " \
                               "WHERE numbits=%d AND " \
                                    "file_type='%s'" % \
                                (config.download.request_numbits, \
                                    config.download.request_datatype.lower()), \
                                fetchone=True)
    if avgrate is None or avgsize is None:
        return min(ALLOWABLE_REQUEST_SIZES)

    # Total number requested that can be downloaded per day (on average).
    max_to_request_per_day = avgrate/avgsize
    
    used = get_space_used()
    avail = get_space_available()
    reserved = get_space_committed()
    
    # Maximum number of bytes that we should request
    max_bytes = min([avail-reserved-config.download.min_free_space, \
                        config.download.space_to_use-reserved-used])
    # Maximum number to request
    max_num = max_bytes/avgsize

    ideal_num_to_request = min([max_num, max_to_request_per_day])

    if debug.DOWNLOAD:
        print "Average dl rate: %.2f bytes/day" % avgrate
        print "Average size per request unit: %d bytes" % avgsize
        print "Max can dl per day: %d" % max_to_request_per_day
        print "Max num to request: %d" % max_num
        print "Ideal to request: %d" % ideal_num_to_request

    # Return the closest allowable request size without exceeding
    # 'ideal_num_to_request'
    num_to_request = max([0]+[N for N in ALLOWABLE_REQUEST_SIZES \
                            if N <= ideal_num_to_request])

    return num_to_request
    


def can_download():
    """Return true if another download can be initiated.
        False otherwise.

        Inputs:
            None
        Output:
            can_dl: A boolean value. True if another download can be
                    initiated. False otherwise.
    """
    downloading = jobtracker.query("SELECT * FROM files " \
                                   "WHERE status='downloading'")
    numdownload = len(downloading)
    used = get_space_used()
    avail = get_space_available()
    
    can_dl = (numdownload < config.download.numdownloads) and \
            (avail > config.download.min_free_space) and \
            (used < config.download.space_to_use)
    return can_dl 


def download(attempt):
    """Given a row from the job-tracker's download_attempts table,
        actually attempt the download.
    """
    file = jobtracker.query("SELECT * FROM files " \
                            "WHERE id=%d" % attempt['file_id'], \
                            fetchone=True)
    request = jobtracker.query("SELECT * FROM requests " \
                               "WHERE id=%d" % file['request_id'], \
                               fetchone=True)

    queries = []

    # Download using bbftp
    res = DownloaderSPAN512.exec_download(request, file)


    # bbftp should report 'get filename OK' if the transfer is successfull
    if res == 'OK': 
        queries.append("UPDATE files " \
                       "SET status='unverified', " \
                            "updated_at='%s', " \
                            "details='Download is complete - File is unverified' " \
                       "WHERE id=%d" % \
                       (jobtracker.nowstr(), file['id']))
        queries.append("UPDATE download_attempts " \
                       "SET status='complete', " \
                            "details='Download is complete', " \
                            "updated_at='%s' " \
                       "WHERE id=%d" % \
                       (jobtracker.nowstr(), attempt['id']))
    else:		       
	queries.append("UPDATE files " \
                       "SET status='failed', " \
                            "updated_at='%s', " \
                            "details='Download failed - %s' " \
                       "WHERE id=%d" % \
                       (jobtracker.nowstr(), str(res), file['id']))
	queries.append("UPDATE download_attempts " \
                       "SET status='download_failed', " \
                            "details='Download failed - %s', " \
                            "updated_at='%s' " \
                       "WHERE id=%d" % \
                       (str(res), jobtracker.nowstr(), attempt['id']))

    jobtracker.query(queries)


def verify_files():
    """For all downloaded files with status 'unverify' verify the files.
        
        Inputs:
            None
        Output:
            numverified: The number of files successfully verified.
    """
    toverify = jobtracker.query("SELECT * FROM files " \
                                "WHERE status='unverified'")

    numverified = 0
    for file in toverify:

        actualsize = pipeline_utils.get_file_size(file['filename'])

        expectedsize = file['size']

        last_attempt_id = jobtracker.query("SELECT id " \
                                           "FROM download_attempts " \
                                           "WHERE file_id=%s " \
                                           "ORDER BY id DESC " % file['id'], \
                                           fetchone=True)
                                                
        queries = []
        if actualsize == expectedsize:
            dlm_cout.outs("Download of %s is complete and verified." % \
                            os.path.split(file['filename'])[-1])
            # Everything checks out!
            queries.append("UPDATE files " \
                           "SET status='downloaded', " \
                                "details='Download is complete and verified', " \
                                "updated_at='%s'" \
                           "WHERE id=%d" % \
                           (jobtracker.nowstr(), file['id']))
            queries.append("UPDATE download_attempts " \
                           "SET status='downloaded', " \
                                "details='Download is complete and verified', " \
                                "updated_at='%s'" \
                           "WHERE id=%d" % \
                           (jobtracker.nowstr(), last_attempt_id))

	    # Mark the beam as downloaded in the main database
	    #mark_beam_downloaded(os.path.split(file['filename'])[-1]))

            numverified += 1
        else:
            dlm_cout.outs("Verification of %s failed. \n" \
                            "\tActual size (%d bytes) != Expected size (%d bytes)" % \
                            (os.path.split(file['filename'])[-1], actualsize, expectedsize))
            
            # Boo... verification failed.
            queries.append("UPDATE files " \
                           "SET status='failed', " \
                                "details='Downloaded file failed verification', " \
                                "updated_at='%s'" \
                           "WHERE id=%d" % \
                           (jobtracker.nowstr(), file['id']))
            queries.append("UPDATE download_attempts " \
                           "SET status='verification_failed', " \
                                "details='Downloaded file failed verification', " \
                                "updated_at='%s'" \
                           "WHERE id=%d" % \
                           (jobtracker.nowstr(), last_attempt_id))
        jobtracker.query(queries)
    return numverified


def recover_failed_downloads():
    """For each entry in the job-tracker DB's files table
        check if the download can be retried or not.
        Update status and clean up, as necessary.
    """
    failed_files = jobtracker.query("SELECT * FROM files " \
                                   "WHERE status='failed'")

    for file in failed_files:
        attempts = jobtracker.query("SELECT * FROM download_attempts " \
                                    "WHERE file_id=%d" % file['id'])
        if len(attempts) < config.download.numretries:
            # download can be retried
            jobtracker.query("UPDATE files " \
                             "SET status='retrying', " \
                                  "updated_at='%s', " \
                                  "details='Download will be attempted again' " \
                             "WHERE id=%s" % \
                             (jobtracker.nowstr(), file['id']))
        else:
            # Abandon this file
            if os.path.exists(file['filename']):
                os.remove(file['filename'])
            jobtracker.query("UPDATE files " \
                             "SET status='terminal_failure', " \
                                  "updated_at='%s', " \
                                  "details='This file has been abandoned' " \
                             "WHERE id=%s" % \
                             (jobtracker.nowstr(), file['id']))

def acknowledge_downloaded_files():
    """Acknowledge the reception of the files
    """
    requests_to_delete = jobtracker.query("SELECT * FROM requests " \
                                          "WHERE status='finished'")
    if len(requests_to_delete) > 0:

        queries = []
        for request_to_delete in requests_to_delete:

            DownloaderSPAN512.delete_stagged_file(request_to_delete)

            dlm_cout.outs("Report download (%s) succeeded." % request_to_delete['guid'])
            queries.append("UPDATE requests " \
                               "SET status='cleaned_up', " \
                               "details='download complete', " \
                               "updated_at='%s' " \
                               "WHERE id=%d" % \
                               (jobtracker.nowstr(), request_to_delete['id']))

        jobtracker.query(queries)
    else: pass


    
def status():
    """Print downloader's status to screen.
    """
    used = get_space_used()
    avail = get_space_available()
    allowed = config.download.space_to_use
    print "Space used by downloaded files: %.2f GB of %.2f GB (%.2f%%)" % \
            (used/1024.0**3, allowed/1024.0**3, 100.0*used/allowed)
    print "Space available on file system: %.2f GB" % (avail/1024.0**3)

    numwait = jobtracker.query("SELECT COUNT(*) FROM requests " \
                               "WHERE status='waiting'", \
                               fetchone=True)
    numfail = jobtracker.query("SELECT COUNT(*) FROM requests " \
                               "WHERE status='failed'", \
                               fetchone=True)
    print "Number of requests waiting: %d" % numwait
    print "Number of failed requests: %d" % numfail

    numdlactive = jobtracker.query("SELECT COUNT(*) FROM files " \
                                   "WHERE status='downloading'", \
                                   fetchone=True)
    numdlfail = jobtracker.query("SELECT COUNT(*) FROM files " \
                                 "WHERE status='failed'", \
                                 fetchone=True)
    print "Number of active downloads: %d" % numdlactive
    print "Number of failed downloads: %d" % numdlfail


def check_downloading_requests():
    requests = jobtracker.query("SELECT * FROM requests "\
                                "WHERE status='downloading'")
    if len(requests) > 0:
        queries = []
        for request in requests:
            files_in_request = jobtracker.query("SELECT * FROM files "\
                                                "WHERE request_id=%d" % \
                                                request['id'])
            downloaded_files = 0
            for f in files_in_request:
                if f['status'] == 'downloaded': downloaded_files += 1
            if downloaded_files == len(files_in_request):
                queries.append("UPDATE requests " \
                               "SET status='finished', " \
                               "details='All files downloaded', " \
                               "updated_at='%s' " \
                               "WHERE id=%d" % \
                               (jobtracker.nowstr(), request['id']))
        jobtracker.query(queries)
    else:
        pass



class DownloadThread(threading.Thread):
    """A sub-class of threading.Thread to download restored
        file from Cornell.
    """
    def __init__(self, attempt):
        """DownloadThread constructor.
            
            Input:
                attempt: A row from the job-tracker's download_attempts table.

            Output:
                self: The DownloadThread object constructed.
        """
        super(DownloadThread, self).__init__(name=attempt['id'])
        self.attempt = attempt

    def run(self):
        """Download data as a separate thread.
        """
        download(self.attempt)
