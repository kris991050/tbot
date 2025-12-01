import sys, os, flask
parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from utils import process_manager


class WebTbotServer():
    def __init__(self, processes:list, port:int=5000, host:str='0.0.0.0'):
        self.processes = processes
        self.port = port
        self.host = host

        self.app = flask.Flask(__name__, template_folder='templates')
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/restart/<string:process_name>', 'restart', self.restart_process)

    def start_web_server(self):
        """Start the Flask server."""
        self.app.run(host=self.host, port=self.port)

    def index(self):
        """Display processes and their statuses in the web interface."""
        processes_info = []
        for process in self.processes:
            status=[]
            for pid in process['pids']:
                proc = process_manager.ProcessUtils.get_process_by('pid', pid)
                status.append(proc[0]['status']) if proc else 'N/A'
            processes_info.append({
                'name': process['name'],
                'pids': process['pids'],
                'status': status  # This should be dynamically updated
            })
        return flask.render_template('index.html', processes=processes_info)

    def restart_process(self, process_name):
        """Handle process restart requests from the web interface."""
        for process in self.processes:
            if process['name'] == process_name:
                process['status'] = 'restarting'  # Just an example
                # Restart the actual process logic
                return flask.redirect(flask.url_for('index'))
        return f"Process {process_name} not found", 404

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
    # Assuming `orchestrator` is an instance of LiveOrchestrator
    orchestrator = LiveOrchestrator(ib, **params)
    app.run(host='0.0.0.0', port=5000)  # Expose Flask on port 5000