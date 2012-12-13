import subprocess
import os

import database
import jobtracker
import config.download

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


def check_request_done(request, dbname='default'):
    """Connect to the data server and check if the request has completed.
        If yes, return True
    """
    db = database.Database(dbname)
    QUERY = "SELECT status FROM full_processing WHERE guid='%s'"%request['guid']
    db.cursor.execute(QUERY)
    status = [row[0] for row in db.cursor.fetchall()]
    db.conn.close()

    # Check if all requested beams are restored (ie in this case, status='restored')
    return all(stat == 'restored' for stat in status)


def get_files_infos(request, dbname='default'):
    """Get the files informations from the data server given the UID
       
        Input:
            request: A row from the requests table.
        Outputs:
            status: A list of path, filename and file size
    """

    db = database.Database(dbname)
    QUERY = "SELECT r.path, r.filename, r.datasize FROM full_processing as p LEFT JOIN stagged_files as r ON r.obs_id=p.obs_id WHERE p.guid='%s'"%request['guid']
    db.cursor.execute(QUERY)
    status = [[os.path.join(row[0],row[1]), row[2]] for row in db.cursor.fetchall()]
    db.conn.close()
    return status


def exec_download(request, file):
    """
    """
    # Download using bbftp
    cmd = "bbftp -e 'setoption localrfio; setoption notmpfile; get %s %s' -u %s %s -E \"/usr/local/bin/bbftpd -s\"" % \
    		(file['remote_filename'], config.download.datadirbbftp, config.commondb.username, config.commondb.host)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, \
            stdin=subprocess.PIPE).stdout.read().strip()
    try:
        res_pipe = pipe.split()[-1]
    except:
        res_pipe = 'Failed'

    return res_pipe


def delete_stagged_file(request, dbname="default"):
    """Given a row from the requests table in the job-tracker DB
	delete a file in the stagging area.

	Input:
            request: A row from the requests table.
	Outputs:
	    None
    """
    db = database.Database(dbname)
    QUERY = "UPDATE full_processing SET status='download complete (to be deleted)', updated_at=NOW() WHERE guid='%s'"%request['guid']
    db.cursor.execute(QUERY)
    db.conn.close()

