
import time
import functools
from rich.table import Table

GRAY_BOLD = "\033[1;30m"
WHITE_BOLD = "\033[1;37m"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
YELLOW_BOLD = "\033[1;33m"
BLUE_BOLD = "\033[1;34m"
MAGENTA_BOLD = "\033[1;35m"
CYAN_BOLD = "\033[1;36m"
DARK_GRAY = "\033[38;5;240m"
GREEN_REVERSE_BOLD = "\033[1;32;7m"
RESET = "\033[0m"
DIVIDER = "─"*64
BOX_TOP = "┌" + "─"*62 + "┐"
BOX_BOTTOM = "└" + "─"*62 + "┘"
BOX_MIDDEL = "│" + " "*62 + "│"
BOX_SIDE = "│"

def log_execution_time(func):
    """
    함수가 실행될 때 시작과 종료 메시지를 출력하고 실행 시간을 측정하는 데코레이터

    Args:
        func (function): 데코레이터를 적용할 함수

    Returns:
        wrapper (function): 데코레이터가 적용된 함수
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{GREEN_REVERSE_BOLD} ∙ Func.{str(func.__name__).upper()} {RESET}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            with open("error.log", "a") as error_file:
                error_file.write(f"[ERROR] Func.{str(func.__name__).upper()}\n")
                error_file.write(f" - Error Message: {str(e)}\n")
                error_file.write(f" - Args: {args}\n")
                error_file.write(f" - Kwargs: {kwargs}\n")
                error_file.write(f" - Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                error_file.write(f"{DIVIDER}\n")
            raise
        end_time = time.time()
        print(f"{DARK_GRAY}Run-Time: {end_time - start_time:.4f}s{RESET}\n")
        return result
    return wrapper

def log_execution_time_with_dist(type: str, func: str, t = None):
    now = time.time()
    if type == "start":
        print(f"{GREEN_REVERSE_BOLD} ∙ Func.{func.upper()} {RESET}")
        return now
    elif type == "end":
        if time is not None:
            print(f"{DARK_GRAY}Run-Time: {now - t:.4f}s{RESET}\n")
        else:
            print(f"{DARK_GRAY}Run-Time: {now:.4f}s{RESET}\n")
        return None
        

def log(cateogry: str, msg: str, upper_div: bool = False, lower_div: bool = False, space: bool = False):
    """
    메시지를 출력하는 함수

    Args:
        category (str): 메시지 카테고리(info, success, warning, error)
        msg (str): 출력할 메시지

    Returns:
        None
    """
    color = None
    if cateogry == "program":
        color = GREEN_BOLD
        msg_length = len(msg)
        gap_left = (62 - msg_length) // 2
        gap_rigth = 62 - msg_length - gap_left
        print(BOX_TOP)
        print(BOX_MIDDEL)
        print(f"{BOX_SIDE}{" "*gap_left}{color}{msg}{RESET}{" "*gap_rigth}{BOX_SIDE}")
        print(BOX_MIDDEL)
        print(BOX_BOTTOM, end="\n")
        return None

    if cateogry == "success":
        color = GREEN_BOLD
        msg = f"✔ {msg}"
    elif cateogry == "warning":
        color = YELLOW_BOLD
    elif cateogry == "error":
        color = RED_BOLD
    elif cateogry == "dimmed":
        color = DARK_GRAY
    else:
        color = WHITE_BOLD
    if upper_div:
        print(DIVIDER)
    print(f"{color}{msg}{RESET}")
    if lower_div:
        print(DIVIDER)
    if space:
        print("\n")
    return None

def log_progress(cur_idx, last_idx, msg):
    """
    진행 상태를 출력하는 함수

    Args:
        cur_idx (int): 현재 인덱스
        last_idx (int): 마지막 인덱스
        msg (str): 출력할 메시지

    Returns:
        None
    """
    frames = ["⠹", "⠼", "⠶", "⠧", "⠏", "⠛"]
    progress = round(((cur_idx) / last_idx * 100), 1)
    cur_frame = frames[cur_idx % len(frames)]
    if cur_idx == last_idx:
        print(f" - [100.0%][{cur_idx}/{last_idx}] {msg} Done!")
    else:
        print(f" {cur_frame} [{progress:5.1f}%][{cur_idx:03d}/{last_idx:03d}] {msg}", end="\r", flush=True)
    return None

def log_summary(summary: dict):
    """
    요약 정보를 출력하는 함수

    Args:
        summary (dict): 요약 정보

    Returns:
        None
    """
    print(f"{WHITE_BOLD}[⌗] Summary{RESET}")
    for key, value in summary.items():
        print(f" - {key}: {value}")
    return None

def console_banner(console):
    console.clear()
    banner = [
        "\n",
        " █████████▙╗   ███▙╗   ▟███╗   ▟██████████╗   ███▙╗   ▟███╗",
        " ███╔═══███║   ████▙╗ ▟████║   ███╔═══════╝   ████▙╗ ▟████║",
        " █████████▍╣   ███▜█▙▟█▛███║   ▜█████████▙╗   ███▜█▙▟█▛███║",
        " ███╔═══███║   ███║▜██▛╔███║           ███║   ███║▜██▛╔███║",
        " ███║   ███║   ███║ ▜▛╔╝███║   ██████████▛║   ███║ ▜▛╔╝███║",
        " ╚══╝   ╚══╝   ╚══╝ ╚═╝ ╚══╝   ╚══════════╝   ╚══╝ ╚═╝ ╚══╝",
        "  R I B  M U S C L E  S E G M E N T A T I O N  M O D U L E ",
        "\n"
    ]
    for line in banner:
        console.print(line, style="bold green")
    
def console_args(console, args):
    table = Table(show_header=True, show_footer=False)
    table.add_column("Argument", width=17)
    table.add_column("Value", width=40)
    for arg, value in args.items():
        table.add_row(arg, str(value))
    console.print(table)
    
def console_classify(console, date_list, sub_list, summary_list):
    table = Table(show_header=True, show_footer=False)
    table.add_column("Date\nSub Dir", width=10)
    for date, sub in zip(date_list, sub_list):
        col = f"{str(date).upper()}\n{str(sub).upper()}"
        table.add_column(col, justify="right")
    for key in summary_list[0].keys():
        row = [key]
        for summary in summary_list:
            row.append(str(summary[key]))
        table.add_row(*row)
    console.print(table)
        
    