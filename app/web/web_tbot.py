import sys, os, flask, psutil, threading, time, multiprocessing
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import process_manager, helpers
from utils.constants import CONSTANTS
from ib_insync import *


class WebTbotServer():
    def __init__(self, pmanager:process_manager.ProcessManager=None, port:int=5000, host:str='127.0.0.1', scan_interval:int=5):
        self.pmanager = pmanager or process_manager.ProcessManager()
        self.port = port
        self.host = host
        self.scan_interval = scan_interval

        self.app = flask.Flask(__name__, template_folder='templates')
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/start/<string:wtype>', 'start_worker', self.start_worker)
        self.app.add_url_rule('/stop/<int:pid>', 'stop_pid', self.stop_pid)

        # Start the background thread for scanning processes
        # self._start_process_scan_thread()
        # self._start_process_scan_multiprocess()

    # def _start_process_scan_multiprocess(self):
    #     """Start a background thread to scan for running processes."""
    #     def scan_processes(queue):
    #         while True:
    #             time.sleep(self.scan_interval)
    #             self._update_process_list(queue)

    #     # Create a manager to share data between processes
    #     manager = multiprocessing.Manager()
    #     queue = manager.Queue()

    #     # Start scanning process in a new process
    #     process = multiprocessing.Process(target=scan_processes, args=(queue,))
    #     process.daemon = True  # Daemonize the process so it terminates when the main process exits
    #     process.start()

    # def _start_process_scan_thread(self, queue=None):
    #     """Start a background thread to scan for running processes."""
    #     def scan_processes():
    #         while True:
    #             time.sleep(self.scan_interval)
    #             self._update_process_list()

    #     # Start scanning process in the background
    #     thread = threading.Thread(target=scan_processes, daemon=True)
    #     thread.start()

    # def _update_process_list(self):
    #     """Update the list of processes in ProcessManager."""
    #     for worker_type in self.pmanager.processes_params.keys():
    #         print(f"🔎 Searching processes for {worker_type}...")
    #         processes_for_worker = process_manager.ProcessUtils.get_process_by('cmdline', [worker_type])#('name', worker_type)
    #         if processes_for_worker:
    #             # If processes are found for this worker type, update the process list
    #             self.pmanager.processes = [process for process in self.pmanager.processes if process['name'] != worker_type]
    #             for process in processes_for_worker:
    #                 self.pmanager.processes.append({
    #                     'name': worker_type,
    #                     'pids': [process['pid']],
    #                     'status': 'running'
    #                 })
    #         # else:
    #         #     # If no processes are found, set the status as 'dead'
    #         #     self.pmanager.processes = [process for process in self.pmanager.processes if process['name'] != worker_type]
    #         #     self.pmanager.processes.append({
    #         #         'name': worker_type,
    #         #         'pids': [],
    #         #         'status': 'dead'
    #         #     })
                
    #     print(f"Checked processes: {self.pmanager.processes}")

    def start_web_server(self):
        """Start the Flask server."""
        self.app.run(host=self.host, port=self.port)
    
    def index(self):
        """Group processes by worker type."""
        grouped = {}

        # Display all workers in the config even if no processes are running
        all_worker_types = self.pmanager.processes_params.keys()  # This gives us all worker types

        for wtype in all_worker_types:
            grouped[wtype] = []

            start_button = {'worker_type': wtype, 'button_label': f"Start New {wtype}", 'url': self.app.url_for('start_worker', wtype=wtype)}

            # Check if any processes are running for this worker type
            processes_for_worker = [p for p in self.pmanager.processes if p['name'] == wtype]
            
            if processes_for_worker:
                # If there are processes running, append them to the group
                for process in processes_for_worker:
                    entries = []
                    for pid in process['pids']:
                        proc = process_manager.ProcessUtils.get_process_by('pid', [pid])
                        status = proc[0]['status'] if proc else 'dead'
                        entries.append({
                            'pid': pid,
                            'status': status,
                            'alive': status == 'running'
                        })
                    grouped[wtype].append(entries)
            else:
                # If no processes are running, show the start button with an empty table
                grouped[wtype] = []
            
            # # Add the start button to the template context
            # grouped[wtype].insert(0, start_button)

        return flask.render_template('index.html', grouped=grouped, all_worker_types=all_worker_types)


    # def index(self):
    #     """Group processes by worker type."""
    #     grouped = {}

    #     for process in self.pmanager.processes:
    #         wtype = process['name']  # pname already equals worker type
    #         if wtype not in grouped:
    #             grouped[wtype] = []

    #         entry = []
    #         for pid in process['pids']:
    #             proc = process_manager.ProcessUtils.get_process_by('pid', [pid])
    #             status = proc[0]['status'] if proc else 'dead'
    #             entry.append({
    #                 'pid': pid,
    #                 'status': status,
    #                 'alive': status == 'running'
    #             })

    #         grouped[wtype].append(entry)

    #     return flask.render_template('index.html', grouped=grouped)
    
    def start_worker(self, wtype):
        """Start a worker using ProcessManager"""
        try:
            # Ensure the worker type is valid
            # if wtype not in ALLOWED_WORKERS:
            #     return f"Error: {wtype} is not a valid worker type.", 400
            wtype = helpers.set_var_with_constraints(wtype, CONSTANTS.WORKERS_TYPES['white'] + CONSTANTS.WORKERS_TYPES['blue'])
            
            # Launch the worker's associated process
            self.pmanager.launch_process(
                target=getattr(process_manager, f"run_live_{wtype}"),
                wtype=wtype
            )
            
            return flask.redirect(flask.url_for('index'))

        except Exception as e:
            return f"Error starting worker {wtype}: {e}", 500
    
    def stop_pid(self, pid):
        """Stop a specific PID"""
        try:
            psutil.Process(pid).terminate()
        except Exception as e:
            return f"Error stopping PID {pid}: {e}", 500

        return flask.redirect(flask.url_for('index'))


    # def restart_process(self, process_name):
    #     """Handle process restart requests from the web interface."""
    #     for process in self.processes:
    #         if process['name'] == process_name:
    #             process['status'] = 'restarting'  # Just an example
    #             # Restart the actual process logic
    #             return flask.redirect(flask.url_for('index'))
    #     return f"Process {process_name} not found", 404


# # Web server routes
# @app.route('/')
# def index():
#     # Display processes and their statuses
#     processes_info = []
#     for process in orchestrator.processes:
#         processes_info.append({
#             'name': process['name'],
#             'pids': process['pids'],
#             'status': 'running' if orchestrator.check_process_alive(process['pids'][0]) else 'stopped'
#         })
#     return flask.render_template('index.html', processes=processes_info)

# @app.route('/restart/<string:process_name>')
# def restart(process_name):
#     """Handle process restart requests."""
#     if orchestrator.stop_process(process_name):
#         return flask.redirect(flask.url_for(self.index_route))
#     else:
#         return f"Process {process_name} not found", 404

if __name__ == '__main__':
    webserver = WebTbotServer()
    webserver.start_web_server()

