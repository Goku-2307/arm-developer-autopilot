from datetime import datetime
import time


class SessionLogger:

    TOTAL_STAGES = 7

    def __init__(self):

        self.events = []

        self.current_stage = 0

        self.session_start = time.perf_counter()

        self.stage_start = None

    def start_stage(self):

        self.stage_start = time.perf_counter()

    def end_stage(self):

        if self.stage_start is None:
            return 0

        duration = time.perf_counter() - self.stage_start

        self.current_stage += 1

        return round(duration, 2)

    def log(self,
            stage,
            message,
            status="INFO",
            duration=None):

        event = {

            "time": datetime.now().strftime("%H:%M:%S"),

            "stage": stage,

            "status": status,

            "message": message,

            "duration": duration

        }

        self.events.append(event)

        text = f"[{event['time']}] "

        text += f"{status:<8}"

        text += f"{stage:<15}"

        text += message

        if duration is not None:

            text += f" ({duration:.2f}s)"

        print(text)

    def info(self, stage, message):

        self.log(stage, message, "INFO")

    def success(self, stage, message):

        duration = self.end_stage()

        self.log(stage,
                 message,
                 "SUCCESS",
                 duration)

    def warning(self, stage, message):

        self.log(stage,
                 message,
                 "WARNING")

    def error(self, stage, message):

        self.log(stage,
                 message,
                 "ERROR")

    def progress(self):

        return min(
            self.current_stage /
            self.TOTAL_STAGES,
            1.0
        )

    def session_time(self):

        return round(

            time.perf_counter() -
            self.session_start,

            2

        )

    def summary(self):

        return {

            "events": len(self.events),

            "completed": self.current_stage,

            "progress": self.progress(),

            "total_time": self.session_time()

        }

    def get_events(self):

        return self.events