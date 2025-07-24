import os
import time
from tabulate import tabulate

def clear_n_line(n):
    """
    Clear the last n lines of the console.
    """
    for _ in range(n):
        print("\033[F\033[K", end='')

patient_list = os.listdir("./data/input")
patient_list.sort()
log_dir = "log"
log_list = os.listdir(log_dir)
log_list.sort()

cur_log = log_list[-1]
cur_log_dir = os.path.join(log_dir, cur_log)

start_time = time.strptime(cur_log, "%Y%m%d_%H%M%S")

log_file_list = os.listdir(os.path.join(log_dir, cur_log))
log_file_list.sort()

waiting_str = "-" + " " * 50

table = [["No", "Patient ID", "Log", "Status"]]

waiting_process_num = 0
done_process_num = 0
            
for idx, patient_id in enumerate(patient_list):
    cur_log_file = os.path.join(cur_log_dir, f"{patient_id}.log")
    if os.path.exists(cur_log_file):
        try:
            with open(cur_log_file, "r") as f:
                lines = f.readlines()
                string = str(lines[-1].strip())
                if "Processing Finished" in string:
                    done_process_num += 1
                    # status = "Done"
                    # table.append([idx, patient_id, "Processing Finished", status])
                else:
                    status = "Processing"
                    table.append([idx, patient_id, string, status])
        except:
            waiting_process_num += 1
            # table.append([idx, patient_id, waiting_str, "Waiting"])
    else:
        waiting_process_num += 1
        # table.append([idx, patient_id, waiting_str, "Waiting"])
        
cur_time = time.localtime()
cur_time_str = time.strftime("%Y-%m-%d %H:%M:%S", cur_time)
        
running_time = int(time.mktime(cur_time) - time.mktime(start_time))
running_time_str = time.strftime("%H:%M:%S", time.gmtime(running_time))

print(f"Current Time: {cur_time_str}, Running Time: {running_time_str} ({running_time}s), Waiting: {waiting_process_num}, Done: {done_process_num}")
print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))