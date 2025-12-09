import os
import datetime
from timeit import default_timer as timer
from lettuce import Reporter, Simulation

def append_txt_file(filename, line: str):
    ''' append a line to a file with an added linebreak'''
    file = open(filename, "a")
    file.write(line + "\n")
    file.close()

class ProgressReporter(Reporter):
    '''
        Progress reporter that logs: current wall time, elapsed wall time,
        elapsed steps and estimates wall time remaining.
        future feature: Option to write a checkpoint file,
                        when t_max is reached.
                        (Sim. can be restarted from checkpoint)

        (!) This reporter does not export other reporters observable values
        etc.,
        so make sure you save them in other ways,
        if sim. is stopped by host system (e.g. HPC cluster)!

    '''

    def __init__(self, interval=1000, t_max=0, i_target=0, i_start=0,
                 outdir=None, print_message=False, checkpoint=False):
        ## initialize local attributes...
        self.t_max = t_max
        self.i_start = i_start
        self.i_target = i_target # should be equivalent to sim.num_steps
        self.outdir= str(outdir)
        self.print_message = print_message
        self.checkpoint = checkpoint

        ## check output directory for file (if path is not NONE)
        # ...OR write to sys.out (print)
        if self.outdir is not None:
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
        self.running = False
        self.t_start = 0
            # mutability of this up for discussion for sims that are
            # ...started from a checkpoint
        self.t_elapsed = 0
            # mutability of this up for discussion for sims that are
            # ...started from a checkpoint

        super().__init__(interval)

    def __call__(self, simulation: 'Simulation'):
        if not self.running:
            self.start_timer()
        elif simulation.flow.i % self.interval == 0:
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%y%m%d_%H%M%S")

            t_now = timer()
            t_elapsed = t_now - self.t_start
            if simulation.flow.i == self.i_start:
                t_per_step = t_elapsed / self.interval
            else:
                t_per_step = t_elapsed / (simulation.flow.i - self.i_start)
            i_remaining = self.i_target - simulation.flow.i
            t_remaining_estimate = t_per_step * i_remaining
            datetime_finish_estimate = timestamp + datetime.timedelta(seconds=t_remaining_estimate)
            t_total_estimate = t_elapsed + t_remaining_estimate

            # write DATA and warn if t_total_estimate > t_max
            base_message = (timestamp_str.ljust(13)
                            + " " + str(simulation.flow.i).rjust(10)
                            + " " + "{:.2f}".format(t_now).rjust(10)
                            + " " + "{:.2f}".format(t_elapsed).rjust(10)
                            + " " + "{:.6f}".format(t_per_step).rjust(10)
                            + " " + "{:.2f}".format(t_remaining_estimate).rjust(15)
                            + " " + "{:.2f}".format(t_total_estimate).rjust(15)
                            + "  " + str(datetime_finish_estimate.strftime(
                                            '%Y-%m-%d %H:%M:%S')).ljust(20))
            # sizes:
            # 13, 10, 7+2.(rjust10), 7+2.(rjust10), 1+6.(rjust10), 7+2.(rjust10), 7+2.(rjust15), 27

            if t_total_estimate > self.t_max:
                message = base_message + " WARNING t_total>t_max=" + str(self.t_max)
            else:
                message = base_message

            append_txt_file(self.outdir + "/progress_reporter_log.txt", message)
            if self.print_message:
                print(message)

            # write checkpoint if t_elapsed > t_max

            if self.checkpoint and self.t_elapsed > self.t_max:
                # checkpointing seems to be missing in current master...
                print("PROGRESS REPORTER: checkpoint was requested, "
                      "but current version of lettuce does not support "
                      "checkpointing... sorry :(")
                # TO BE IMPLEMENTED IN THE FUTURE:
                # simulation.save_checkpoint(self.outdir
                #   + "/" + timestamp_str + "_f_"
                #   + str(simulation.flow.i) + ".cpt")

    def start_timer(self):
        self.running = True
        self.t_start = timer()
        #print("starting timer")
        print("-> PROGRESS_REPORTER ACTIVE:\nt_start: " + str(self.t_start)
              + ", interval: " + str(self.interval) + ", i_target: " + str(self.i_target))
        
        table_header = ("timestamp ".center(13)
                        +"|"+"step".center(10)
                        +"|"+"t_now".center(10)
                        +"|"+"t_elapsed".center(10)
                        +"|"+"t_per_step".center(10)
                        +"|"+"t_remain(est)".center(15)
                        +"|"+"t_total(est)".center(15)
                        +"|"+"DATE_FINISH(est)".center(20)
                        +"|"+" T WARNING")
        if self.print_message:
            print(table_header)
        else:
            print(f"print_message == False: see "
                  f"'{self.outdir}/progress_reporter_log.txt' for output")

        append_txt_file(self.outdir+"/progress_reporter_log.txt",
                        "t_start: "+str(self.t_start)+", interval: "+str(self.interval)
                        +", i_target: "+str(self.i_target))
        append_txt_file(self.outdir+"/progress_reporter_log.txt", table_header)
        # sizes: 13, 10, 7+2.(rjust10), 7+2.(rjust10), 1+6.(rjust10), 7+2.(rjust10), 7+2.(rjust15), 27